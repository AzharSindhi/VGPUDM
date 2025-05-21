# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import transforms as pth_transforms
# from . import vision_transformer as vits
from pointnet2.models import vision_transformer as vits

class DinoEncoder:
    def __init__(self, image_size=480, arch = 'vit_small', patch_size = 8, aggregate="class", device="cuda") -> None:
        # convert args to init parameters


        self.device = torch.device(device)
        self.patch_size = patch_size
        # build model
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)
        self.normalize_transform = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            assert False, "There is no reference weights available for this model => We use random weights."

        self.aggregate = aggregate
        self.model = model
        self.out_dim = 384 #model.embed_dim
        self.to_pil_test = pth_transforms.ToPILImage()

    def get_image_features(self, img):

        # transform = pth_transforms.Compose([
        #     pth_transforms.Resize(image_size),
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        
        img = self.normalize_transform(img)
        # img_test = self.to_pil_test(img[0])
        # img_test.save("test.png")
        # image has already batch dimension
        # make the image divisible by the patch size
        w, h = img.shape[2] - img.shape[2] % self.patch_size, img.shape[3] - img.shape[3] % self.patch_size
        img = img[:, :, :w, :h]
        
        attentions = self.model.get_intermediate_layers(img.to(self.device), n=1)
        class_token = None
        if self.aggregate == "class":
            # get the class token embedding
            class_token = attentions[0][:, 0, :]
        elif self.aggregate == "avgpool":
            # average pooling across second dim
            output = torch.cat([x[:, 0] for x in attentions], dim=-1)
            output = torch.cat((output.unsqueeze(-1), torch.mean(attentions[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            class_token = output.reshape(output.shape[0], -1)
        elif self.aggregate == "none":
            class_token = torch.cat(attentions, dim=-1)
        else:
            raise ValueError("Invalid aggregate type")

        return class_token

def get_images_from_pc(point_cloud_file, outdir):
    import numpy as np
    import open3d as o3d
    from PIL import Image
    import os # For checking file existence

    if "gt" not in point_cloud_file:
        return
        
    if not os.path.exists(point_cloud_file):
        print(f"Warning: Point cloud file '{point_cloud_file}' not found.")
        print("Using a randomly generated sphere as a fallback point cloud.")
        # Create a fallback sphere point cloud if the specified file doesn't exist
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        pcd_fallback = mesh.sample_points_poisson_disk(number_of_points=2048)
        pc_numpy = np.asarray(pcd_fallback.points)
    else:
        pc_numpy = np.loadtxt(point_cloud_file)
    
    # Downsample the point cloud (optional, but good for dense clouds)
    # The original code used 'npoints = 256'. You can adjust this.
    # More points will result in a denser image but slower processing.
    npoints_subsample = 2048 # Number of points to use for rendering
    if len(pc_numpy) > npoints_subsample:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_numpy)
        # Farthest point downsampling helps maintain overall shape
        pcd_down = pcd.farthest_point_down_sample(npoints_subsample)
        points_to_render = np.asarray(pcd_down.points)
    else:
        points_to_render = pc_numpy

    # --- Configuration for rendering the binary image ---
    output_image_width = 512  # Desired width of the binary image
    output_image_height = 512 # Desired height of the binary image
    point_draw_size = 2       # Size of the dot for each point in pixels (e.g., 1 for single pixel, 2 for 2x2, etc.)
    background_color = 0      # Pixel value for background (typically black)
    foreground_color = 255    # Pixel value for points (typically white)

    # --- Point Cloud Preprocessing for Projection ---
    if points_to_render.shape[0] == 0:
        print("Point cloud is empty. Creating a blank image.")
        binary_image_array = np.full((output_image_height, output_image_width), background_color, dtype=np.uint8)
    else:
        # 1. Center the point cloud around the origin
        mean_coords = np.mean(points_to_render, axis=0)
        pc_centered = points_to_render - mean_coords

        # 2. Normalize the point cloud to fit within a [-1, 1] cube for a canonical view.
        #    This ensures the object is visible regardless of its original scale or position.
        max_abs_coord = np.max(np.abs(pc_centered))
        if max_abs_coord == 0:  # Avoid division by zero if it's a single point at origin or all points are identical
            max_abs_coord = 1.0
        pc_normalized = pc_centered / max_abs_coord
        # pc_normalized now has coordinates roughly in [-1, 1] for x, y, and z.

        # --- Render to Binary Image with Z-buffering (for occlusion) ---
        # Initialize image array (e.g., all black)
        binary_image_array = np.full((output_image_height, output_image_width), background_color, dtype=np.uint8)
        
        # Initialize Z-buffer. Stores the depth of the closest point found so far for each pixel.
        # We assume the camera is looking along the -Z axis (from +Z towards the origin).
        # Therefore, points with larger Z values in pc_normalized are closer to the camera.
        # Initialize z_buffer with negative infinity, as larger Z means closer.
        z_buffer = np.full((output_image_height, output_image_width), -np.inf, dtype=np.float32)

        # Orthographic projection:
        # X-coordinate of point maps to image column (u)
        # Y-coordinate of point maps to image row (v) (with Y inversion for image conventions)
        # Z-coordinate of point is used for the depth test

        for i in range(pc_normalized.shape[0]):
            x, y, z = pc_normalized[i, 0], pc_normalized[i, 1], pc_normalized[i, 2]

            # Convert normalized coordinates to image pixel coordinates:
            # Map x from [-1, 1] to [0, output_image_width - 1]
            # Map y from [-1, 1] to [0, output_image_height - 1]
            # Image coordinate system: (0,0) is usually top-left.
            # For y: y_norm = 1 (top of point cloud in normalized space) should map to v = 0 (top row of image).
            #          y_norm = -1 (bottom of point cloud) should map to v = output_image_height - 1 (bottom row).
            # This means: v = ( (1 - y_norm) / 2 ) * (height - 1)
            
            img_u_center = int(((x + 1.0) / 2.0) * (output_image_width - 1))
            img_v_center = int(((1.0 - y) / 2.0) * (output_image_height - 1)) # Y is inverted for image coords

            # Draw a small square for each point for better visibility
            # Calculate the bounds of the square to draw for the current point
            u_min_draw = img_u_center - (point_draw_size - 1) // 2
            u_max_draw = img_u_center + point_draw_size // 2 
            v_min_draw = img_v_center - (point_draw_size - 1) // 2
            v_max_draw = img_v_center + point_draw_size // 2

            # Iterate over the pixels in this square footprint
            for current_v_pixel in range(max(0, v_min_draw), min(output_image_height, v_max_draw + 1)):
                for current_u_pixel in range(max(0, u_min_draw), min(output_image_width, u_max_draw + 1)):
                    # Z-buffer check: if this point (z) is closer than what's already at this pixel
                    if z > z_buffer[current_v_pixel, current_u_pixel]:
                        z_buffer[current_v_pixel, current_u_pixel] = z
                        binary_image_array[current_v_pixel, current_u_pixel] = foreground_color
    
    # Convert the NumPy array to a PIL Image
    binary_image_pil = Image.fromarray(binary_image_array, mode='L') # 'L' mode for grayscale (binary)
    
    # Save the resulting binary image
    output_filename = os.path.basename(point_cloud_file).replace(".xyz", ".png")
    binary_image_pil.save(os.path.join(outdir, output_filename))
    print(f"Saved '{os.path.join(outdir, output_filename)}' ({output_image_width}x{output_image_height})")

    # --- Optional: Prepare and use the DinoEncoder with the generated image ---
    # If you want to feed this binary image into your DinoEncoder:
    use_dino_encoder = False # Set to True to run this part
    if use_dino_encoder:
        print("\nPreparing image for DINO Encoder...")
        # 1. DinoEncoder typically expects a 3-channel RGB image.
        #    Convert the binary (single-channel) image to 3-channel by replicating the channel.
        image_for_dino_rgb_np = np.stack([binary_image_array]*3, axis=-1)
        image_for_dino_pil = Image.fromarray(image_for_dino_rgb_np, mode='RGB')
        
        # 2. The DinoEncoder has its own image size and normalization.
        dino_image_size = 480 # Default from DinoEncoder class, or pass as arg
        
        # Create a transform pipeline for DINO input
        # Note: The DinoEncoder's self.normalize_transform is applied *inside* its get_image_features method.
        # So, we only need to resize and convert to tensor here.
        transform_for_dino = pth_transforms.Compose([
            pth_transforms.Resize((dino_image_size, dino_image_size)), # DINO models often expect square images
            pth_transforms.ToTensor(),
            # The internal normalize_transform will be applied by the encoder
        ])
        
        img_tensor_for_dino = transform_for_dino(image_for_dino_pil)
        if img_tensor_for_dino.shape[0] == 1: # If ToTensor makes it (1, H, W) for grayscale
            img_tensor_for_dino = img_tensor_for_dino.repeat(3,1,1) # Convert to (3, H, W)
        img_tensor_for_dino = img_tensor_for_dino.unsqueeze(0) # Add batch dimension -> (1, 3, H, W)

        print(f"Input tensor shape for DINO: {img_tensor_for_dino.shape}")

        # Initialize and use the encoder
        # Ensure you have downloaded the DINO weights or handled the case where they are not found.
        try:
            dino_enc = DinoEncoder(image_size=dino_image_size, device="cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad(): # Ensure no gradients are computed during inference
                 features = dino_enc.get_image_features(img_tensor_for_dino)
            print(f"DINO features shape: {features.shape}") # Example: (1, 384) for vit_small
        except Exception as e:
            print(f"Error during DINO encoding: {e}")
            print("This might be due to missing pretrained weights or an issue with the model setup.")

if __name__ == "__main__":
    import os
    point_cloud_dir = "/home/hpc/iwnt/iwnt150h/VGPUDM/pointnet2/exp_vipc/clip_ViPC_only_clip/vis/"
    outdir = "/home/hpc/iwnt/iwnt150h/VGPUDM/pointnet2/exp_vipc/clip_ViPC_only_clip/out_images/"
    os.makedirs(outdir, exist_ok=True)
    pointcloud_files = os.listdir(point_cloud_dir)
    for point_cloud_file in pointcloud_files:
        get_images_from_pc(point_cloud_file, outdir)

    # # read pointcloud from .xyz
    # import numpy as np 
    # import open3d as o3d
    # import torch
    # from PIL import Image
    # pc = np.loadtxt("/home/hpc/iwnt/iwnt150h/VGPUDM/pointnet2/example/pig.xyz")
    # pc = torch.from_numpy(pc).float()
    # npoints = 256
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # pcd_down = pcd.farthest_point_down_sample(npoints)
    # pc_down = np.asarray(pcd_down.points)


    # # normalize the pc_down between 0 and 1
    # pc_down = (pc_down - pc_down.min(axis=0)) / (pc_down.max(axis=0) - pc_down.min(axis=0))
    # pc_down = np.repeat(pc_down, npoints, axis=0).reshape(npoints, npoints, 3)
    # # drop the z dimension
    # pc_down = pc_down[:, :, :1]
    # pc_down = (pc_down * 255).astype(np.uint8)
    # # binarize pc_down based on uniform sampling of 80% of points
    # #TO DO
    
    # # convert to PIL image
    # binary_image = Image.fromarray(pc_down)
    # binary_image.save("binary_image.png")
    
    # import numpy as np
    # import open3d as o3d
    # from PIL import Image
    # import os # To check for file existence

    # # --- Configuration ---
    # # This matches 'npoints' from your original code and will define
    # # the number of points sampled and the dimensions of the output image.
    # outdir = "/home/hpc/iwnt/iwnt150h/VGPUDM/pointnet2/exp_vipc/clip_ViPC_only_clip/test_images/"
    # os.makedirs(outdir, exist_ok=True)

    # point_cloud_filepath = "/home/hpc/iwnt/iwnt150h/VGPUDM/pointnet2/exp_vipc/clip_ViPC_only_clip/vis/02691156_5fed73635306ad9f14ac58bc87dcf2c2_17_gt.xyz"


    # npoints_param = 1024
    # output_image_width = npoints_param
    # output_image_height = npoints_param
    # background_color = 0  # Black
    # foreground_color = 255 # White

    # # Update this path to your .xyz file

    # # --- Load Point Cloud ---
    # if not os.path.exists(point_cloud_filepath):
    #     print(f"Warning: File '{point_cloud_filepath}' not found. Using a random 1000-point cloud as a fallback.")
    #     # Create a dummy point cloud if the specified file isn't found
    #     pc_data_np = np.random.rand(1000, 3) * 100 # Random points in a 100x100x100 cube
    # else:
    #     pc_data_np = np.loadtxt(point_cloud_filepath)

    # # --- Downsample Point Cloud ---
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc_data_np)
    # # Farthest point sampling to get a representative subset of 'npoints_param'
    # pcd_down = pcd.farthest_point_down_sample(npoints_param)
    # sampled_points = np.asarray(pcd_down.points)  # Shape: (npoints_param or less, 3)

    # # --- Create the Binary Image ---
    # # 1. Initialize a blank image array (all background_color)
    # image_array = np.full((output_image_height, output_image_width), background_color, dtype=np.uint8)

    # if sampled_points.shape[0] > 0:
    #     # 2. Isolate X and Y coordinates for projection
    #     #    Z coordinates (sampled_points[:, 2]) are ignored in this simple 2D projection
    #     x_coords = sampled_points[:, 0]
    #     y_coords = sampled_points[:, 1]

    #     # 3. Normalize X coordinates to fit image width [0, output_image_width - 1]
    #     x_min, x_max = x_coords.min(), x_coords.max()
    #     if (x_max - x_min) < 1e-6: # Avoid division by zero or if all points have same X
    #         img_x_coords = np.full_like(x_coords, (output_image_width - 1) // 2, dtype=int)
    #     else:
    #         img_x_coords = ((x_coords - x_min) / (x_max - x_min) * (output_image_width - 1)).astype(int)

    #     # 4. Normalize Y coordinates to fit image height [0, output_image_height - 1]
    #     #    Also, invert Y-axis: point cloud's min_y -> image bottom, max_y -> image top
    #     y_min, y_max = y_coords.min(), y_coords.max()
    #     if (y_max - y_min) < 1e-6: # Avoid division by zero or if all points have same Y
    #         img_y_coords = np.full_like(y_coords, (output_image_height - 1) // 2, dtype=int)
    #     else:
    #         # normalized_y maps [y_min, y_max] to [0, 1]
    #         normalized_y = (y_coords - y_min) / (y_max - y_min)
    #         # Invert and scale: maps normalized 0 (y_min) to (height-1), and 1 (y_max) to 0
    #         img_y_coords = ((1.0 - normalized_y) * (output_image_height - 1)).astype(int)

    #     # 5. "Draw" points onto the image_array by setting pixels to foreground_color
    #     # Clip coordinates to ensure they are within the image bounds before assignment
    #     img_x_coords = np.clip(img_x_coords, 0, output_image_width - 1)
    #     img_y_coords = np.clip(img_y_coords, 0, output_image_height - 1)
        
    #     # Use advanced indexing to set multiple points at once
    #     image_array[img_y_coords, img_x_coords] = foreground_color

    # # Optional: Interpretation of "# binarize pc_down based on uniform sampling of 80% of points"
    # # If you mean to only draw 80% of the sampled_points:
    # apply_80_percent_point_sampling = False  # Set to True to activate this behavior
    # if apply_80_percent_point_sampling and sampled_points.shape[0] > 0:
    #     print("Applying 80% point sampling before drawing.")
    #     image_array.fill(background_color) # Reset image to black
        
    #     num_points_to_actually_draw = int(0.95 * sampled_points.shape[0])
    #     if num_points_to_actually_draw > 0 :
    #         # Randomly choose indices of the points to draw
    #         indices_to_draw = np.random.choice(sampled_points.shape[0], num_points_to_actually_draw, replace=False)
            
    #         # Get the image coordinates for these chosen points
    #         sampled_img_x_coords = img_x_coords[indices_to_draw]
    #         sampled_img_y_coords = img_y_coords[indices_to_draw]
            
    #         # Draw only these sampled points
    #         image_array[sampled_img_y_coords, sampled_img_x_coords] = foreground_color

    # # --- Convert to PIL Image and Save ---
    # # The `image_array` is 2D (height, width). For a single-channel grayscale/binary image,
    # # PIL's 'L' mode expects this 2D format.
    # binary_image = Image.fromarray(image_array, mode='L')

    # output_filename = "binary_image_from_points.png"
    # binary_image.save(output_filename)
    # print(f"Saved '{output_filename}' ({output_image_width}x{output_image_height})")
    pass


