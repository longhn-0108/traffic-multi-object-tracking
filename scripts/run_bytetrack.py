"""CLI wrapper for YOLO + ByteTrack."""
from mot_tracking.bytetrack_tracker import parse_args, run_tracking, run_tracking_folder

if __name__ == "__main__":
    args = parse_args()
    if args.single:
        run_tracking(
            output_path=args.output,
            show=args.show,
            device=args.device,
            video_path=args.input_video,
            conf=args.conf,
            iou=args.iou,
            image_scale=args.scale,
            save_pred_csv=args.save_pred_csv,
            save_eval_csv=args.save_eval_csv,
            save_trackeval_txt=args.save_trackeval_txt,
            max_frames=args.max_frames,
            write_video=not args.no_video,
            save_csv=not args.no_csv,
            tracker_yaml=args.tracker_yaml,
            model_path=args.model,
            stitch_ids=not args.no_stitch_ids,
            stitch_max_gap=args.stitch_max_gap,
            stitch_iou=args.stitch_iou,
            stitch_center_dist=args.stitch_center_dist,
        )
    else:
        run_tracking_folder(
            input_dir=args.input_dir,
            video_output_dir=args.video_output_dir,
            trackeval_tracker_dir=args.trackeval_tracker_dir,
            show=args.show,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_frames=args.max_frames,
            write_video=not args.no_video,
            save_csv=not args.no_csv,
            tracker_yaml=args.tracker_yaml,
            model_path=args.model,
            stitch_ids=not args.no_stitch_ids,
            stitch_max_gap=args.stitch_max_gap,
            stitch_iou=args.stitch_iou,
            stitch_center_dist=args.stitch_center_dist,
        )
