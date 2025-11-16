import torch
import pathlib
import imageio

from trellis.representations.gaussian import Gaussian
from trellis.utils import render_utils

dimension = (4, 3, 1)
biggest_dimension = max(dimension)
# shrink_transform = [[1 / biggest_dimension, 0, 0], [0, 1 / biggest_dimension, 0], [0, 0, 1 / biggest_dimension]]
transform = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
# transform = torch.tensor(flip_transform, dtype=torch.float) @ torch.tensor(shrink_transform, dtype=torch.float) # shrink then flip
# transform = transform.numpy().tolist()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=pathlib.Path, required=True)
    parser.add_argument('--num_frames', type=int, default=300)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--r', type=float, default=2)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--only_gaussians_scene', action='store_true')  # only process files named "gaussians_scene.ply", if set, also there will be no transform
    args = parser.parse_args()
    
    ply_files = list(args.input_directory.rglob('*.ply'))
    
    # only include files named "gaussians_scene.ply"
    if args.only_gaussians_scene:
        ply_files = [f for f in ply_files if f.name == 'gaussians_scene.ply']
        transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # no transform
    
    print(f'Found {len(ply_files)} .ply files in {args.input_directory}')
    
    def process(path: pathlib.Path):
        gaussian = Gaussian(aabb=[-1, -1, -1, 1, 1, 1])
        gaussian.load_ply(path, transform=transform)
        video = render_utils.render_video(gaussian, num_frames=args.num_frames, resolution=args.resolution, r=args.r, bg_color=(255, 255, 255), pitch_mean=0.3, pitch_offset=0.0)['color']

        save_path = path.with_suffix('.close.mp4')
        if save_path.exists():
            save_path.unlink()
        imageio.mimsave(save_path, video, fps=30)
    
    for path in ply_files:
        if args.skip_existing and path.with_suffix('.close.mp4').exists():
            print(f'Skipping existing {path.with_suffix(".close.mp4")}')
            continue
        
        if args.dry:
            print(f'Would process {path}')
        else:
            print(f'Processing {path}')
            process(path)