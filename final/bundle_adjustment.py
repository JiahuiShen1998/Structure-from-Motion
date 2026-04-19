import copy

import pycolmap
from pycolmap import logging


def solve_ba(reconstruction, ba_options, ba_config):
    """
    Solve the global or local BA problem.

    Args:
        reconstruction (pycolmap.Reconstruction): Scene reconstruction object.
        ba_options (pycolmap.BundleAdjustmentOptions): BA options config.
        ba_config (pycolmap.BundleAdjustmentConfig): BA config specifying cameras/points.

    Returns:
        pycolmap.SolverSummary: Summary of the BA solving process.
    """
    bundle_adjuster = pycolmap.create_default_bundle_adjuster(
        ba_options, ba_config, reconstruction
    )
    summary = bundle_adjuster.solve()
    return summary


def run_global_ba(mapper, mapper_options, ba_options):
    """
    Run global bundle adjustment.

    Args:
        mapper (pycolmap.IncrementalMapper): Incremental mapper instance.
        mapper_options (pycolmap.MapperOptions): Mapper options.
        ba_options (pycolmap.BundleAdjustmentOptions): BA options.
    """
    reconstruction = mapper.reconstruction
    assert reconstruction is not None, "No valid reconstruction found."

    reg_image_ids = reconstruction.reg_image_ids()
    if len(reg_image_ids) < 2:
        logging.fatal("At least two images must be registered for global BA.")

    # Make a copy of the original BA options for potential local changes
    ba_options_tmp = copy.deepcopy(ba_options)

    # If the registered images are too few, use stricter convergence parameters
    if len(reg_image_ids) < 10:  # kMinNumRegImagesForFastBA = 10
        ba_options_tmp.solver_options.function_tolerance /= 10
        ba_options_tmp.solver_options.gradient_tolerance /= 10
        ba_options_tmp.solver_options.parameter_tolerance /= 10
        ba_options_tmp.solver_options.max_num_iterations *= 2
        ba_options_tmp.solver_options.max_linear_solver_iterations = 200

    # Filter negative depth observations to avoid degeneracy
    mapper.observation_manager.filter_observations_with_negative_depth()

    # Setup BA config
    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in reg_image_ids:
        ba_config.add_image(image_id)

    # Fix poses if needed (e.g., for existing images)
    if mapper_options.fix_existing_images:
        for image_id in reg_image_ids:
            if image_id in mapper.existing_image_ids:
                ba_config.set_constant_cam_pose(image_id)

    # Fix 7-DOFs: anchor the first two images
    reg_image_ids_iter = iter(reg_image_ids)
    first_image_id = next(reg_image_ids_iter)
    second_image_id = next(reg_image_ids_iter)
    ba_config.set_constant_cam_pose(first_image_id)
    if (not mapper_options.fix_existing_images) or (
        second_image_id not in mapper.existing_image_ids
    ):
        ba_config.set_constant_cam_positions(second_image_id, [0])

    # Solve BA
    summary = solve_ba(reconstruction, ba_options_tmp, ba_config)
    logging.info("Global Bundle Adjustment")
    logging.info(summary.BriefReport())


def refine_global_iteratively(
    mapper,
    max_num_refinements,
    max_refinement_change,
    mapper_options,
    ba_options,
    tri_options,
    normalize_reconstruction=True,
):
    """
    Perform iterative global refinements with BA.

    Args:
        mapper (pycolmap.IncrementalMapper): Incremental mapper instance.
        max_num_refinements (int): Max refinement iterations.
        max_refinement_change (float): Stopping threshold for change ratio.
        mapper_options (pycolmap.MapperOptions): Mapper options.
        ba_options (pycolmap.BundleAdjustmentOptions): BA options.
        tri_options (pycolmap.TriangulationOptions): Triangulation options.
        normalize_reconstruction (bool): If True, normalize after each BA.
    """
    reconstruction = mapper.reconstruction

    # Merge track info and retriangulate
    mapper.complete_and_merge_tracks(tri_options)
    num_retriangulated_obs = mapper.retriangulate(tri_options)
    logging.verbose(1, f"=> Retriangulated observations: {num_retriangulated_obs}")

    for _ in range(max_num_refinements):
        # Record current observation count
        num_obs = reconstruction.compute_num_observations()

        # Run global BA
        run_global_ba(mapper, mapper_options, ba_options)

        # Optionally normalize the reconstruction
        if normalize_reconstruction:
            reconstruction.normalize()

        # Complete tracks, then filter outliers
        num_changed = mapper.complete_and_merge_tracks(tri_options)
        num_changed += mapper.filter_points(mapper_options)

        ratio_changed = (num_changed / num_obs) if num_obs > 0 else 0
        logging.verbose(1, f"=> Changed observations: {ratio_changed:.6f}")

        if ratio_changed < max_refinement_change:
            break


def run_local_ba(mapper, mapper_options, ba_options, tri_options, image_id, point3D_ids):
    """
    Local BA / Run local bundle adjustment around one image and its neighbors.

    Args:
        mapper (pycolmap.IncrementalMapper): The incremental mapper instance.
        mapper_options (pycolmap.MapperOptions): Mapper options.
        ba_options (pycolmap.BundleAdjustmentOptions): BA options.
        tri_options (pycolmap.TriangulationOptions): Triangulation options.
        image_id (int): Target image ID around which local BA is performed.
        point3D_ids (list[int]): List of 3D point IDs to possibly refine.

    Returns:
        pycolmap.LocalBundleAdjustmentReport: Report of local BA results.
    """
    reconstruction = mapper.reconstruction
    assert reconstruction is not None, "No valid reconstruction found."

    report = pycolmap.LocalBundleAdjustmentReport()

    # Find local bundle (neighboring images)
    local_bundle = mapper.find_local_bundle(mapper_options, image_id)

    # Only perform local BA if there's any local context
    if local_bundle:
        ba_config = pycolmap.BundleAdjustmentConfig()
        ba_config.add_image(image_id)
        for local_img_id in local_bundle:
            ba_config.add_image(local_img_id)

        # Fix poses if they are part of existing images
        if mapper_options.fix_existing_images:
            for local_img_id in local_bundle:
                if local_img_id in mapper.existing_image_ids:
                    ba_config.set_constant_cam_pose(local_img_id)

        # Fix camera intrinsics if part of a larger set
        num_images_per_cam = {}
        for img_id in ba_config.image_ids:
            img = reconstruction.images[img_id]
            num_images_per_cam[img.camera_id] = (
                num_images_per_cam.get(img.camera_id, 0) + 1
            )
        for cam_id, count_local in num_images_per_cam.items():
            if count_local < mapper.num_reg_images_per_camera[cam_id]:
                ba_config.set_constant_cam_intrinsics(cam_id)

        # Fix 7 DOFs
        if len(local_bundle) == 1:
            ba_config.set_constant_cam_pose(local_bundle[0])
            ba_config.set_constant_cam_positions(image_id, [0])
        elif len(local_bundle) > 1:
            image_id1, image_id2 = local_bundle[-1], local_bundle[-2]
            ba_config.set_constant_cam_pose(image_id1)
            if (not mapper_options.fix_existing_images) or (
                image_id2 not in mapper.existing_image_ids
            ):
                ba_config.set_constant_cam_positions(image_id2, [0])

        # Add short-track or newly created points as variables
        variable_point3D_ids = set()
        for pid in list(point3D_ids):
            p3D = reconstruction.point3D(pid)
            kMaxTrackLength = 15
            if p3D.error == -1.0 or p3D.track.length() <= kMaxTrackLength:
                ba_config.add_variable_point(pid)
                variable_point3D_ids.add(pid)

        # Solve local BA
        summary = solve_ba(mapper.reconstruction, ba_options, ba_config)
        logging.info("Local Bundle Adjustment")
        logging.info(summary.BriefReport())

        report.num_adjusted_observations = int(summary.num_residuals / 2)

        # Merge newly refined tracks
        report.num_merged_observations = mapper.triangulator.merge_tracks(
            tri_options, variable_point3D_ids
        )
        # Complete tracks after pose refinement
        report.num_completed_observations = mapper.triangulator.complete_tracks(
            tri_options, variable_point3D_ids
        )
        report.num_completed_observations += mapper.triangulator.complete_image(
            tri_options, image_id
        )

    # Filter outliers
    filter_image_ids = {image_id, *local_bundle}
    report.num_filtered_observations = mapper.observation_manager.filter_points3D_in_images(
        mapper_options.filter_max_reproj_error,
        mapper_options.filter_min_tri_angle,
        filter_image_ids,
    )
    report.num_filtered_observations += mapper.observation_manager.filter_points3D(
        mapper_options.filter_max_reproj_error,
        mapper_options.filter_min_tri_angle,
        point3D_ids,
    )

    return report


def refine_local_iteratively(
    mapper,
    max_num_refinements,
    max_refinement_change,
    mapper_options,
    ba_options,
    tri_options,
    image_id,
):
    """
    Perform iterative local refinements with BA.

    Args:
        mapper (pycolmap.IncrementalMapper): Incremental mapper instance.
        max_num_refinements (int): Maximum number of refinement iterations.
        max_refinement_change (float): Stopping threshold for change ratio.
        mapper_options (pycolmap.MapperOptions): Mapper options.
        ba_options (pycolmap.BundleAdjustmentOptions): BA options.
        tri_options (pycolmap.TriangulationOptions): Triangulation options.
        image_id (int): Image ID around which to refine.
    """
    ba_options_tmp = copy.deepcopy(ba_options)

    for _ in range(max_num_refinements):
        # Run local BA on the target image and its neighbors
        report = run_local_ba(
            mapper,
            mapper_options,
            ba_options_tmp,
            tri_options,
            image_id,
            mapper.get_modified_points3D(),
        )

        logging.verbose(1, f"=> Merged observations: {report.num_merged_observations}")
        logging.verbose(1, f"=> Completed observations: {report.num_completed_observations}")
        logging.verbose(1, f"=> Filtered observations: {report.num_filtered_observations}")

        changed = 0
        if report.num_adjusted_observations > 0:
            changed = (
                report.num_merged_observations
                + report.num_completed_observations
                + report.num_filtered_observations
            ) / report.num_adjusted_observations
        logging.verbose(1, f"=> Changed observations: {changed:.6f}")

        if changed < max_refinement_change:
            break

        # After the first iteration, switch to a trivial loss function 
        ba_options_tmp.loss_function_type = pycolmap.LossFunctionType.TRIVIAL

    # Clear the modified points after local refinement
    mapper.clear_modified_points3D()
