"""
Python reimplementation of the C++ incremental mapper with equivalent logic.

"""

import argparse
import time
from pathlib import Path

import enlighten
import pycolmap
from pycolmap import logging

# Custom modules
import bundle_adjustment


def extract_image_colors(image_path, image_id, reconstruction):
    """
    Extract color info from specified image into reconstruction.

    Args:
        image_path (str): The path to the image folder.
        image_id (int): The ID of the image to extract colors for.
        reconstruction (pycolmap.Reconstruction): The reconstruction object.
    """
    if not reconstruction.extract_colors_for_image(image_id, image_path):
        logging.warning(f"Could not read image {image_id} at path {image_path}")


def save_reconstruction_snapshot(reconstruction, snapshot_path):
    """
    保存当前重建快照 / Save current reconstruction snapshot.

    Args:
        reconstruction (pycolmap.Reconstruction): Reconstruction object.
        snapshot_path (pathlib.Path): Folder path to save snapshot.
    """
    logging.info("Creating snapshot...")
    timestamp = time.time() * 1000
    path = snapshot_path / f"{timestamp:010d}"
    path.mkdir(exist_ok=True, parents=True)
    logging.verbose(1, f"=> Writing snapshot to: {path}")
    reconstruction.write(path)


def global_refinement_loop(options, mapper_options, mapper):
    """
    全局精细化步骤，包括重新三角化和全局BA / 
    Perform global retriangulation and global bundle adjustment iteratively.

    Args:
        options (pycolmap.IncrementalPipelineOptions): Pipeline options.
        mapper_options (pycolmap.MapperOptions): Mapper configuration.
        mapper (pycolmap.IncrementalMapper): The incremental mapper instance.
    """
    logging.info("Retriangulation and Global bundle adjustment...")
    bundle_adjustment.refine_global_iteratively(
        mapper,
        options.ba_global_max_refinements,
        options.ba_global_max_refinement_change,
        mapper_options,
        options.get_global_bundle_adjustment(),
        options.get_triangulation(),
    )
    mapper.filter_images(mapper_options)


def init_reconstruction(controller, mapper, mapper_options, reconstruction):
    """
    初始化重建，寻找合适的初始图像对，并进行全局BA / 
    Initialize reconstruction by finding a good initial pair & performing global BA.

    Equivalent to IncrementalPipeline.initialize_reconstruction(...) in pycolmap.

    Args:
        controller (pycolmap.IncrementalPipeline): Pipeline controller with options.
        mapper (pycolmap.IncrementalMapper): The incremental mapper instance.
        mapper_options (pycolmap.MapperOptions): Mapper options.
        reconstruction (pycolmap.Reconstruction): Reconstruction object.

    Returns:
        pycolmap.IncrementalMapperStatus: The status of initialization.
    """
    options = controller.options
    init_pair = (options.init_image_id1, options.init_image_id2)

    # 若没有指定初始图像对，则自动寻找 / If no init pair provided, find one
    if not options.is_initial_pair_provided():
        logging.info("Finding good initial image pair...")
        ret = mapper.find_initial_image_pair(mapper_options, *init_pair)
        if ret is None:
            logging.info("No good initial image pair found.")
            return pycolmap.IncrementalMapperStatus.NO_INITIAL_PAIR
        init_pair, two_view_geometry = ret
    else:
        # 如果用户指定了初始对，但在重建中并不存在，直接返回错误 / If user-provided pair doesn't exist
        if not all(reconstruction.exists_image(i) for i in init_pair):
            logging.info("=> Pair does not exist.")
            return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR
        two_view_geometry = mapper.estimate_initial_two_view_geometry(
            mapper_options, *init_pair
        )
        if two_view_geometry is None:
            logging.info("Provided pair is insuitable.")
            return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR

    logging.info(f"Initializing with image pair {init_pair}")
    mapper.register_initial_image_pair(mapper_options, two_view_geometry, *init_pair)
    logging.info("Global bundle adjustment...")
    bundle_adjustment.run_global_ba(
        mapper, mapper_options, options.get_global_bundle_adjustment()
    )
    reconstruction.normalize()
    mapper.filter_points(mapper_options)
    mapper.filter_images(mapper_options)

    if reconstruction.num_reg_images() == 0 or reconstruction.num_points3D() == 0:
        return pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR

    if options.extract_colors:
        extract_image_colors(controller.image_path, init_pair[0], reconstruction)

    return pycolmap.IncrementalMapperStatus.SUCCESS


def reconstruct_single_model(controller, mapper, mapper_options, reconstruction):
    """
    递增式重建子模型的逻辑 / Reconstruct a single sub-model incrementally.

    Equivalent to IncrementalPipeline.reconstruct_sub_model(...)

    Args:
        controller (pycolmap.IncrementalPipeline): Pipeline controller with options.
        mapper (pycolmap.IncrementalMapper): Incremental mapper instance.
        mapper_options (pycolmap.MapperOptions): Mapper configuration.
        reconstruction (pycolmap.Reconstruction): Reconstruction object.

    Returns:
        pycolmap.IncrementalMapperStatus: Reconstruction status.
    """
    mapper.begin_reconstruction(reconstruction)

    # 如尚未注册任何图像，则先进行初始化 / If no registered images, initialize
    if reconstruction.num_reg_images() == 0:
        init_status = init_reconstruction(controller, mapper, mapper_options, reconstruction)
        if init_status != pycolmap.IncrementalMapperStatus.SUCCESS:
            return init_status
        controller.callback(pycolmap.IncrementalMapperCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK)

    options = controller.options
    snapshot_prev_num_reg_images = reconstruction.num_reg_images()
    ba_prev_num_reg_images = reconstruction.num_reg_images()
    ba_prev_num_points = reconstruction.num_points3D()

    reg_next_success, prev_reg_next_success = True, True
    while True:
        # 如果连续两次都注册失败，则退出 / If registration fails consecutively, break
        if not (reg_next_success or prev_reg_next_success):
            break
        prev_reg_next_success = reg_next_success
        reg_next_success = False

        next_images = mapper.find_next_images(mapper_options)
        if len(next_images) == 0:
            break

        for reg_trial in range(len(next_images)):
            next_image_id = next_images[reg_trial]
            logging.info(
                f"Registering image #{next_image_id}, "
                f"(resistered image num: {reconstruction.num_reg_images() + 1})"
            )
            num_vis = mapper.observation_manager.num_visible_points3D(next_image_id)
            num_obs = mapper.observation_manager.num_observations(next_image_id)
            logging.info(f"=> Visible points: {num_vis} / {num_obs}")

            reg_next_success = mapper.register_next_image(mapper_options, next_image_id)
            if reg_next_success:
                break
            else:
                logging.info("=> Could not register, trying another.")

    
            # If initial pair fails too many times, might abort this pair
            kMinNumInitialRegTrials = 30
            if reg_trial >= kMinNumInitialRegTrials and reconstruction.num_reg_images() < options.min_model_size:
                break

        if reg_next_success:
            mapper.triangulate_image(options.get_triangulation(), next_image_id)
            # Equivalent to iterative_local_refinement
            bundle_adjustment.refine_local_iteratively(
                mapper,
                options.ba_local_max_refinements,
                options.ba_local_max_refinement_change,
                mapper_options,
                options.get_local_bundle_adjustment(),
                options.get_triangulation(),
                next_image_id,
            )

            # Check if global refinement is needed
            if controller.check_run_global_refinement(
                reconstruction, ba_prev_num_reg_images, ba_prev_num_points
            ):
                global_refinement_loop(options, mapper_options, mapper)
                ba_prev_num_points = reconstruction.num_points3D()
                ba_prev_num_reg_images = reconstruction.num_reg_images()

            if options.extract_colors:
                extract_image_colors(controller.image_path, next_image_id, reconstruction)

            # Snapshot saving
            if (
                options.snapshot_images_freq > 0
                and reconstruction.num_reg_images() >= options.snapshot_images_freq + snapshot_prev_num_reg_images
            ):
                snapshot_prev_num_reg_images = reconstruction.num_reg_images()
                save_reconstruction_snapshot(reconstruction, Path(options.snapshot_path))

            controller.callback(pycolmap.IncrementalMapperCallback.NEXT_IMAGE_REG_CALLBACK)

        # If max overlap reached, break
        if mapper.num_shared_reg_images() >= int(options.max_model_overlap):
            break

        # If registration fails, try global refinement
        if (not reg_next_success) and prev_reg_next_success:
            global_refinement_loop(options, mapper_options, mapper)

    # Final global refinement if needed
    if (
        reconstruction.num_reg_images() >= 2
        and reconstruction.num_reg_images() != ba_prev_num_reg_images
        and reconstruction.num_points3D() != ba_prev_num_points
    ):
        global_refinement_loop(options, mapper_options, mapper)

    return pycolmap.IncrementalMapperStatus.SUCCESS


def reconstruct_incrementally(controller, mapper_options):
    """
    Perform incremental reconstruction on images from DB.

    Equivalent to IncrementalPipeline.reconstruct(...)

    Args:
        controller (pycolmap.IncrementalPipeline): Pipeline controller with options.
        mapper_options (pycolmap.MapperOptions): Mapper configuration.
    """
    options = controller.options
    reconstruction_manager = controller.reconstruction_manager
    database_cache = controller.database_cache

    mapper = pycolmap.IncrementalMapper(database_cache)
    initial_reconstruction_given = reconstruction_manager.size() > 0

    if reconstruction_manager.size() > 1:
        logging.fatal("Can only resume from a single reconstruction if multiple are given.")

    for num_trials in range(options.init_num_trials):
        if (not initial_reconstruction_given) or num_trials > 0:
            reconstruction_idx = reconstruction_manager.add()
        else:
            reconstruction_idx = 0

        reconstruction = reconstruction_manager.get(reconstruction_idx)
        status = reconstruct_single_model(controller, mapper, mapper_options, reconstruction)

        if status == pycolmap.IncrementalMapperStatus.INTERRUPTED:
            mapper.end_reconstruction(False)
        elif status in (
            pycolmap.IncrementalMapperStatus.NO_INITIAL_PAIR,
            pycolmap.IncrementalMapperStatus.BAD_INITIAL_PAIR,
        ):
            mapper.end_reconstruction(True)
            reconstruction_manager.delete(reconstruction_idx)
            if options.is_initial_pair_provided():
                return
        elif status == pycolmap.IncrementalMapperStatus.SUCCESS:
            total_num_reg_images = mapper.num_total_reg_images()
            min_model_size = min(
                0.8 * database_cache.num_images(), options.min_model_size
            )
            # If the number of registered images is less than the threshold, the current reconstruction is deleted (in multi-model scenarios)
            if (
                options.multiple_models
                and reconstruction_manager.size() > 1
                and (
                    reconstruction.num_reg_images() < min_model_size
                    or reconstruction.num_reg_images() == 0
                )
            ):
                mapper.end_reconstruction(True)
                reconstruction_manager.delete(reconstruction_idx)
            else:
                mapper.end_reconstruction(False)

            controller.callback(pycolmap.IncrementalMapperCallback.LAST_IMAGE_REG_CALLBACK)

            if (
                initial_reconstruction_given
                or (not options.multiple_models)
                or reconstruction_manager.size() >= options.max_num_models
                or total_num_reg_images >= database_cache.num_images() - 1
            ):
                return
        else:
            logging.fatal(f"Unknown reconstruction status: {status}")


def incremental_mapper_pipeline(controller):
    """
    The main incremental mapper pipeline.

    Equivalent to IncrementalPipeline.run() in pycolmap.

    Args:
        controller (pycolmap.IncrementalPipeline): The pipeline controller.
    """
    timer = pycolmap.Timer()
    timer.start()

    # Load DB into controller
    if not controller.load_database():
        return

    init_mapper_options = controller.options.get_mapper()
    reconstruct_incrementally(controller, init_mapper_options)

    # Relax constraints if no reconstruction is found
    for _ in range(2):
        if controller.reconstruction_manager.size() > 0:
            break
        logging.info("=> Relaxing the initialization constraints...")
        init_mapper_options.init_min_num_inliers //= 2
        reconstruct_incrementally(controller, init_mapper_options)
        if controller.reconstruction_manager.size() > 0:
            break
        logging.info("=> Further relaxing initialization constraints...")
        init_mapper_options.init_min_tri_angle /= 2
        reconstruct_incrementally(controller, init_mapper_options)

    timer.print_minutes()


def run_incremental_sfm(
    database_path,
    image_path,
    output_path,
    options=None,
    input_path=None,
):
    """
    Run the incremental SfM pipeline.

    Args:
        database_path (Path): Path to the COLMAP-compatible database.
        image_path (Path): Path to the images folder.
        output_path (Path): Path to save reconstructions.
        options (pycolmap.IncrementalPipelineOptions): Pipeline configuration.
        input_path (Path): (Optional) Path to an existing reconstruction.

    Returns:
        dict: A dictionary of reconstructions indexed by an integer.
    """
    if options is None:
        options = pycolmap.IncrementalPipelineOptions()

    if not database_path.exists():
        logging.fatal(f"Database path does not exist: {database_path}")
    if not image_path.exists():
        logging.fatal(f"Image path does not exist: {image_path}")

    output_path.mkdir(exist_ok=True, parents=True)
    reconstruction_manager = pycolmap.ReconstructionManager()

    if input_path is not None and input_path != "":
        reconstruction_manager.read(input_path)

    # Initialize the incremental pipeline
    mapper = pycolmap.IncrementalPipeline(
        options, image_path, database_path, reconstruction_manager
    )

    # Use enlighten manager for progress feedback
    num_images = pycolmap.Database(database_path).num_images
    with enlighten.Manager() as manager:
        with manager.counter(total=num_images, desc="Images registered:") as pbar:
            pbar.update(0, force=True)
            mapper.add_callback(
                pycolmap.IncrementalMapperCallback.INITIAL_IMAGE_PAIR_REG_CALLBACK,
                lambda: pbar.update(2),
            )
            mapper.add_callback(
                pycolmap.IncrementalMapperCallback.NEXT_IMAGE_REG_CALLBACK,
                lambda: pbar.update(1),
            )
            incremental_mapper_pipeline(mapper)

    # Write reconstructions & return
    reconstruction_manager.write(output_path)
    reconstructions = {}
    for i in range(reconstruction_manager.size()):
        reconstructions[i] = reconstruction_manager.get(i)
    return reconstructions


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True, help="Path to the COLMAP database.")
    parser.add_argument("--image_path", required=True, help="Path to the image folder.")
    parser.add_argument("--input_path", default=None, help="(Optional) Path to existing reconstructions.")
    parser.add_argument("--output_path", required=True, help="Folder to save new reconstructions.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_incremental_sfm(
        database_path=Path(args.database_path),
        image_path=Path(args.image_path),
        input_path=Path(args.input_path) if args.input_path else None,
        output_path=Path(args.output_path),
    )
