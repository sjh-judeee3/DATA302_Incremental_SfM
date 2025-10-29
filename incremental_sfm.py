import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import os
from collections import defaultdict
from scipy.spatial.transform import Rotation

class IncrementalSfM:
    def __init__(self, focal_length=800):
        """
        Initialize SfM pipeline
        
        Arguments:
            focal_length: Camera focal length estimate (default: 800)
        """
        self.images = []
        self.cameras = {}  # {image_idx: {'K': K, 'R': R, 't': t}}
        self.points_3d = []  # List of 3D points
        self.point_colors = []  # RGB colors for each 3D point
        self.observations = {}  # {point_idx: {image_idx: keypoint_idx}}
        self.features = []  # Store features for each image
        self.matches = {}  # Store matches between image pairs
        self.focal_length = focal_length
        
    def extract_frames(self, video_path, frame_skip=10, max_frames=50, output_dir='output/images'):
        """
        Extract frames from video and save them to dedicated output_dir
        
        Arguments:
            video_path: Path to video file
            frame_skip: Extract every n-th frame
            max_frames: Maximum number of frames to extract
            output_dir: Directory to save extracted frames
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        frame_idx = 0
        extracted = 0
        
        print("=" * 60)
        print("STEP 1: Frame Extraction")
        print("=" * 60)
        print(f"  Saving frames to: {os.path.abspath(output_dir)}")
        
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip == 0:
                image_filename = f"frame_{extracted:04d}.jpg"
                image_path = os.path.join(output_dir, image_filename)
                cv2.imwrite(image_path, frame) 

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                self.images.append({
                    'idx': len(self.images),
                    'image': gray,
                    'color': frame,
                    'shape': gray.shape,
                    'path': image_path  
                })
                extracted += 1
                
            frame_idx += 1
        
        cap.release()
        print(f"Extracted and saved {len(self.images)} frames")
        print(f"Frame size: {self.images[0]['shape']}")
        print()
        
        if len(self.images) < 2:
            raise ValueError("Not enough frames extracted. Need at least 2 frames.")
    
    def detect_and_match_features(self):
        """
        Detect features in all images and match between pairs
        """
        print("=" * 60)
        print("STEP 2: Feature Detection and Matching")
        print("=" * 60)
        
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures=2000)
        
        # Detect features in all images
        print("Detecting features...")
        for i, img_data in enumerate(self.images):
            kp, desc = sift.detectAndCompute(img_data['image'], None)
            self.features.append({
                'keypoints': kp,
                'descriptors': desc
            })
            print(f"  Image {i}: {len(kp)} keypoints")
        
        # Match features between all pairs
        print("\nMatching features between image pairs...")
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        for i in range(len(self.images)):
            for j in range(i+1, len(self.images)):
                if self.features[i]['descriptors'] is None or \
                   self.features[j]['descriptors'] is None:
                    continue
                
                # KNN matching with k=2 for ratio test
                raw_matches = bf.knnMatch(
                    self.features[i]['descriptors'],
                    self.features[j]['descriptors'],
                    k=2
                )
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in raw_matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                if len(good_matches) > 30:  # Minimum matches threshold
                    self.matches[(i, j)] = good_matches
                    print(f"  Images {i}-{j}: {len(good_matches)} matches")
        
        print(f"\nFound {len(self.matches)} valid image pairs")
        print()
    
    def get_camera_matrix(self, image_idx):
        """
        Get K: Camera intrinsic matrix 
        
        Arguments:
            image_idx: Index of image
            
        Returns:
            K: 3x3 camera intrinsic matrix
        """
        h, w = self.images[image_idx]['shape']
        K = np.array([
            [self.focal_length, 0, w/2],
            [0, self.focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        return K
    
    def select_initial_pair(self):
        """
        Select best initial image pair
        
        Criteria:
        - Many feature matches
        - Good baseline (not too similar views)
        - High inlier ratio with essential matrix
        
        Returns:
            (img1_idx, img2_idx): Indices of best pair
        """
        print("=" * 60)
        print("STEP 3: Initial Pair Selection")
        print("=" * 60)
        
        best_pair = None
        best_score = 0
        best_inliers = 0
        
        K = self.get_camera_matrix(0)
        
        for (i, j), match_list in self.matches.items():
            # Get matched points
            pts1 = np.float32([
                self.features[i]['keypoints'][m.queryIdx].pt 
                for m in match_list
            ])
            pts2 = np.float32([
                self.features[j]['keypoints'][m.trainIdx].pt 
                for m in match_list
            ])
            
            # Estimate essential matrix with RANSAC
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            
            if E is None or mask is None:
                continue
            
            inliers = mask.sum()
            inlier_ratio = inliers / len(match_list)
            
            # Score combines inlier count and ratio
            score = inliers * inlier_ratio
            
            if score > best_score and inliers > 50:
                best_score = score
                best_pair = (i, j)
                best_inliers = inliers
        
        if best_pair is None:
            raise ValueError("Could not find suitable initial pair")
        
        print(f"Selected initial pair: Images {best_pair[0]} - {best_pair[1]}")
        print(f"  Matches: {len(self.matches[best_pair])}")
        print(f"  Inliers: {best_inliers}")
        print()
        
        return best_pair
    
    def initialize_reconstruction(self):
        """
        Initialize reconstruction with first two views
        """
        print("=" * 60)
        print("STEP 4: Initial Reconstruction (Two Views)")
        print("=" * 60)
        
        # Select best initial pair
        img1_idx, img2_idx = self.select_initial_pair()
        match_list = self.matches[(img1_idx, img2_idx)]
        
        # Get matched points
        pts1 = np.float32([
            self.features[img1_idx]['keypoints'][m.queryIdx].pt 
            for m in match_list
        ])
        pts2 = np.float32([
            self.features[img2_idx]['keypoints'][m.trainIdx].pt 
            for m in match_list
        ])
        
        # Camera intrinsic matrix
        K = self.get_camera_matrix(img1_idx)
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        # Recover pose (R, t) from essential matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        
        # First camera at origin
        self.cameras[img1_idx] = {
            'K': K,
            'R': np.eye(3, dtype=np.float64),
            't': np.zeros((3, 1), dtype=np.float64)
        }
        
        # Second camera with relative pose
        self.cameras[img2_idx] = {
            'K': K,
            'R': R,
            't': t
        }
        
        # Triangulate initial 3D points
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        # Filter valid points (in front of both cameras, reasonable depth)
        valid_count = 0
        for i, point in enumerate(points_3d):
            if not mask_pose[i]:
                continue
            
            # Check if point is in front of both cameras
            # For first camera (at origin)
            point_cam1 = self.cameras[img1_idx]['R'] @ point + self.cameras[img1_idx]['t'].flatten()
            if point_cam1[2] <= 0:
                continue
            
            # For second camera
            point_cam2 = R @ point + t.flatten()
            if point_cam2[2] <= 0:
                continue
            
            # Check reasonable depth (not too far)
            if point_cam1[2] > 100 or point_cam2[2] > 100:
                continue
            
            # Add point
            point_idx = len(self.points_3d)
            self.points_3d.append(point)
            
            # Get color from first image
            x, y = pts1[i].astype(int)
            h, w = self.images[img1_idx]['shape']
            x = np.clip(x, 0, w-1)
            y = np.clip(y, 0, h-1)
            color = self.images[img1_idx]['color'][y, x]
            self.point_colors.append(color)
            
            # Record observations
            self.observations[point_idx] = {
                img1_idx: match_list[i].queryIdx,
                img2_idx: match_list[i].trainIdx
            }
            valid_count += 1
        
        print(f"Initial reconstruction complete")
        print(f"  Registered cameras: 2")
        print(f"  Triangulated 3D points: {len(self.points_3d)}")
        print()
        
        if len(self.points_3d) < 10:
            raise ValueError("Too few initial 3D points. Try different video or parameters.")
    
    def select_next_view(self):
        """
        Select next best view to add
        
        Criteria:
        - See many already reconstructed 3D points
        - Have good matches with registered views
        
        Returns:
            next_idx: Index of next image to register
            visible_points: Number of visible 3D points
        """
        registered = set(self.cameras.keys())
        unregistered = set(img['idx'] for img in self.images if 'failed_registration' not in img) - registered

        if not unregistered:
            return None, 0
        
        best_view = None
        max_visible = 0
        
        for img_idx in unregistered:
            visible_count = 0
            
            # Count how many 3D points are potentially visible
            for point_idx, obs in self.observations.items():
                for reg_idx in obs.keys():
                    if reg_idx not in registered:
                        continue
                    
                    pair = tuple(sorted([img_idx, reg_idx]))
                    if pair in self.matches:
                        # Check if this 3D point's keypoint is in the match list
                        reg_kp_idx = obs[reg_idx]
                        for m in self.matches[pair]:
                            if (pair[0] == reg_idx and m.queryIdx == reg_kp_idx) or \
                               (pair[1] == reg_idx and m.trainIdx == reg_kp_idx):
                                visible_count += 1
                                break
                        break # Move to next 3D point
            
            if visible_count > max_visible:
                max_visible = visible_count
                best_view = img_idx
        
        return best_view, max_visible
    
    def register_camera(self, img_idx):
        """
        Register new camera using PnP
        
        Arguments:
            img_idx: Index of image to register
            
        Returns:
            success: True if registration successful
        """
        # Find 2D-3D correspondences
        points_3d = []
        points_2d = []
        point_indices = []
        
        registered = set(self.cameras.keys())
        
        for point_idx, obs in self.observations.items():
            # Check if this 3D point is visible in both:
            # 1. A registered camera
            # 2. The new camera we're trying to register
            
            for reg_idx in obs.keys():
                if reg_idx not in registered:
                    continue
                
                # Check if we have matches between img_idx and reg_idx
                pair = tuple(sorted([img_idx, reg_idx]))
                if pair not in self.matches:
                    continue
                
                # Find the matching keypoint in img_idx
                match_list = self.matches[pair]
                reg_kp_idx = obs[reg_idx]
                
                for m in match_list:
                    # Check which index corresponds to which image
                    if pair[0] == reg_idx:
                        if m.queryIdx == reg_kp_idx:
                            new_kp_idx = m.trainIdx
                            break
                    else: # pair[1] == reg_idx
                        if m.trainIdx == reg_kp_idx:
                            new_kp_idx = m.queryIdx
                            break
                else:
                    continue # No match found for this reg_idx
                
                # Found correspondence
                points_3d.append(self.points_3d[point_idx])
                kp = self.features[img_idx]['keypoints'][new_kp_idx]
                points_2d.append(kp.pt)
                point_indices.append((point_idx, new_kp_idx))
                break # Found a 2D-3D correspondence, move to next 3D point
        
        if len(points_3d) < 6:
            print(f"Not enough 2D-3D correspondences: {len(points_3d)}")
            return False
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        
        # Solve PnP with RANSAC
        K = self.get_camera_matrix(img_idx)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None,
            iterationsCount=1000,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if not success or inliers is None or len(inliers) < 6:
            print(f"solvePnPRansac failed (inliers: {len(inliers) if inliers is not None else 0})")
            return False
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Store camera
        self.cameras[img_idx] = {
            'K': K,
            'R': R,
            't': tvec
        }
        
        # Update observations for inlier points
        for idx in inliers.flatten():
            point_idx, kp_idx = point_indices[idx]
            self.observations[point_idx][img_idx] = kp_idx
        
        print(f"PnP successful with {len(inliers)} inliers")
        return True
    
    def triangulate_new_points(self, new_img_idx):
        """
        Triangulate new 3D points visible in newly registered camera
        
        Args:
            new_img_idx: Index of newly registered camera
        """
        registered = set(self.cameras.keys())
        new_points = 0
        
        # Compare new camera with all previously registered cameras
        for reg_idx in registered:
            if reg_idx == new_img_idx:
                continue
            
            pair = tuple(sorted([new_img_idx, reg_idx]))
            if pair not in self.matches:
                continue
            
            match_list = self.matches[pair]
            
            # Get camera matrices
            K = self.get_camera_matrix(new_img_idx) # K is same for all
            R1 = self.cameras[reg_idx]['R']
            t1 = self.cameras[reg_idx]['t']
            R2 = self.cameras[new_img_idx]['R']
            t2 = self.cameras[new_img_idx]['t']
            
            P1 = K @ np.hstack([R1, t1])
            P2 = K @ np.hstack([R2, t2])
            
            for m in match_list:
                # Check if already triangulated
                already_exists = False
                if pair[0] == reg_idx:
                    kp1_idx = m.queryIdx
                    kp2_idx = m.trainIdx
                else:
                    kp1_idx = m.trainIdx
                    kp2_idx = m.queryIdx
                
                # Check if point from reg_idx is already in observations
                for obs in self.observations.values():
                    if reg_idx in obs and obs[reg_idx] == kp1_idx:
                        already_exists = True
                        break
                
                if already_exists:
                    continue
                
                # Triangulate
                kp1 = self.features[reg_idx]['keypoints'][kp1_idx]
                kp2 = self.features[new_img_idx]['keypoints'][kp2_idx]
                
                pts1 = np.array([kp1.pt], dtype=np.float32).T
                pts2 = np.array([kp2.pt], dtype=np.float32).T
                
                point_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
                point_3d = (point_4d[:3] / point_4d[3]).flatten()
                
                # Validate point
                # Check in front of first camera
                point_cam1 = R1 @ point_3d + t1.flatten()
                if point_cam1[2] <= 0 or point_cam1[2] > 100:
                    continue
                
                # Check in front of second camera
                point_cam2 = R2 @ point_3d + t2.flatten()
                if point_cam2[2] <= 0 or point_cam2[2] > 100:
                    continue
                
                # Add point
                point_idx = len(self.points_3d)
                self.points_3d.append(point_3d)
                
                # Get color
                x, y = kp1.pt
                x, y = int(x), int(y)
                h, w = self.images[reg_idx]['shape']
                x = np.clip(x, 0, w-1)
                y = np.clip(y, 0, h-1)
                color = self.images[reg_idx]['color'][y, x]
                self.point_colors.append(color)
                
                # Record observation
                self.observations[point_idx] = {
                    reg_idx: kp1_idx,
                    new_img_idx: kp2_idx
                }
                new_points += 1
        
        return new_points
    
    def incremental_reconstruction(self):
        """
        Incrementally add cameras and triangulate points
        """
        print("=" * 60)
        print("STEP 5: Incremental Reconstruction")
        print("=" * 60)
        
        iteration = 0
        while len(self.cameras) < len(self.images):
            iteration += 1
            
            # Select next view
            next_view, visible = self.select_next_view()
            
            if next_view is None or visible < 20: # Increased threshold
                print(f"\nNo more views to register (iteration {iteration})")
                print(f"  (Best view {next_view} has {visible} potential matches)")
                break
            
            print(f"\nIteration {iteration}:")
            print(f"  Attempting to register image {next_view}")
            print(f"  Potential 2D-3D matches: {visible}")
            
            # Register camera
            success = self.register_camera(next_view)
            
            if not success:
                print(f"Failed to register camera {next_view}")
                # Mark as failed
                self.images[next_view]['failed_registration'] = True 
                continue
            
            print(f"Camera {next_view} registered successfully")
            
            # Triangulate new points
            new_points = self.triangulate_new_points(next_view)
            print(f"Triangulated {new_points} new 3D points")
            
            print(f"  Status: {len(self.cameras)}/{len(self.images)} cameras, "
                  f"{len(self.points_3d)} 3D points")
        
        print(f"\n{'=' * 60}")
        print("RECONSTRUCTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Registered cameras: {len(self.cameras)}/{len(self.images)}")
        print(f"Reconstructed 3D points: {len(self.points_3d)}")
        print()

    def bundle_adjustment(self):
        """
        Optimize cameras and points with Bundle Adjustment
        """
        
        # --- Helper function for residuals ---
        def residuals(params, n_cameras, n_points, camera_indices, observations, features, K):
            
            # Unpack parameters
            camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
            point_params = params[n_cameras * 6:].reshape((n_points, 3))
            
            errors = []
            
            for point_idx, obs in observations.items():
                if point_idx >= n_points: continue # Safety check
                point_3d = point_params[point_idx]
                
                for img_idx, kp_idx in obs.items():
                    if img_idx not in camera_indices:
                        continue
                        
                    cam_idx = camera_indices[img_idx]
                    rvec = camera_params[cam_idx, :3]
                    tvec = camera_params[cam_idx, 3:]
                    
                    # Project 3D point onto image
                    projected, _ = cv2.projectPoints(
                        point_3d.reshape(1, 1, 3), # Needs shape (N, 1, 3)
                        rvec, tvec, K, None
                    )
                    projected = projected.reshape(2)
                    
                    # Get observed 2D keypoint
                    kp = features[img_idx]['keypoints'][kp_idx]
                    observed = np.array(kp.pt)
                    
                    # Calculate reprojection error
                    error = projected - observed
                    errors.extend(error)
            
            return np.array(errors)
        # --- End of helper function ---

        
        n_cameras = len(self.cameras)
        n_points = len(self.points_3d)
        
        if n_points == 0 or n_cameras == 0:
            print("Cannot run BA: No points or cameras.")
            return

        # Map from global image_idx to local BA camera index (0 to n_cameras-1)
        camera_indices = {idx: i for i, idx in enumerate(self.cameras.keys())}
        
        # Pack initial parameters into one vector
        x0 = []
        
        # Add camera parameters (rvec, tvec)
        for img_idx in self.cameras.keys():
            cam = self.cameras[img_idx]
            rvec, _ = cv2.Rodrigues(cam['R'])
            x0.extend(rvec.flatten())
            x0.extend(cam['t'].flatten())
        
        # Add 3D point parameters
        x0.extend(np.array(self.points_3d).flatten())
        
        x0 = np.array(x0)
        
        # Get a K matrix (assumed constant)
        K = self.get_camera_matrix(list(self.cameras.keys())[0])
        
        # Run optimization
        result = least_squares(
            residuals, 
            x0, 
            verbose=2, 
            method='trf', # Trust Region Reflective
            ftol=1e-4,
            args=(n_cameras, n_points, camera_indices, self.observations, self.features, K)
        )
        
        # Unpack optimized results
        optimized = result.x
        camera_params = optimized[:n_cameras * 6].reshape((n_cameras, 6))
        point_params = optimized[n_cameras * 6:].reshape((n_points, 3))
        
        # Update cameras
        for img_idx, local_idx in camera_indices.items():
            R, _ = cv2.Rodrigues(camera_params[local_idx, :3])
            t = camera_params[local_idx, 3:].reshape(3, 1)
            self.cameras[img_idx]['R'] = R
            self.cameras[img_idx]['t'] = t
        
        # Update 3D points
        self.points_3d = [p for p in point_params]

    
    def visualize(self, save_path='reconstruction.png'):
        """
        Visualize reconstructed cameras and 3D points
        
        Args:
            save_path: Path to save visualization
        """
        print("=" * 60)
        print("STEP 6: Visualization")
        print("=" * 60)
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D points
        if len(self.points_3d) > 0:
            points = np.array(self.points_3d)
            colors = np.array(self.point_colors) / 255.0
            
            # Subsample if too many points
            if len(points) > 10000:
                indices = np.random.choice(len(points), 10000, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            # OpenCV BGR -> Matplotlib RGB
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=colors[:, ::-1], 
                      marker='.', s=1, alpha=0.5)
        
        # Plot cameras
        camera_positions = []
        for idx, cam in self.cameras.items():
            R = cam['R']
            t = cam['t'].flatten()
            
            # Camera center in world coordinates
            C = -R.T @ t
            camera_positions.append(C)
            
            # Plot camera position
            ax.scatter(C[0], C[1], C[2], c='red', marker='o', s=100)
            
            # Plot camera orientation
            direction = R.T @ np.array([0, 0, 1])
            ax.quiver(C[0], C[1], C[2],
                     direction[0], direction[1], direction[2],
                     length=0.5, color='blue', arrow_length_ratio=0.3)
            
            # Add label
            ax.text(C[0], C[1], C[2], f'  {idx}', fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'SfM Reconstruction\n'
                    f'{len(self.cameras)} cameras, {len(self.points_3d)} 3D points',
                    fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        if camera_positions:
            camera_positions = np.array(camera_positions)
            max_range = np.array([
                camera_positions[:, 0].max() - camera_positions[:, 0].min(),
                camera_positions[:, 1].max() - camera_positions[:, 1].min(),
                camera_positions[:, 2].max() - camera_positions[:, 2].min()
            ]).max() / 2.0
            
            # Avoid division by zero if max_range is 0
            if max_range < 1e-6:
                max_range = 1.0

            mid_x = (camera_positions[:, 0].max() + camera_positions[:, 0].min()) * 0.5
            mid_y = (camera_positions[:, 1].max() + camera_positions[:, 1].min()) * 0.5
            mid_z = (camera_positions[:, 2].max() + camera_positions[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    def save_results(self, output_dir='output'):
        """
        Save reconstruction results
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cameras
        with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
            f.write("# Camera parameters\n")
            f.write("# image_idx fx fy cx cy qw qx qy qz tx ty tz\n")
            for idx, cam in self.cameras.items():
                K = cam['K']
                R = cam['R']
                t = cam['t'].flatten()
                
                # Convert rotation matrix to quaternion
                quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
                quat = np.roll(quat, 1)  # [w, x, y, z]
                
                f.write(f"{idx} {K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f} "
                       f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
                       f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")
        
        # Save points
        with open(os.path.join(output_dir, 'points.txt'), 'w') as f:
            f.write("# 3D points\n")
            f.write("# X Y Z R G B\n")
            for point, color in zip(self.points_3d, self.point_colors):
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                       f"{int(color[2])} {int(color[1])} {int(color[0])}\n") # BGR -> RGB
        
        print(f"Saved results (cameras.txt, points.txt) to: {output_dir}/")
    
    def run(self, video_path, frame_skip=10, max_frames=30, base_output_dir='output'):
        """
        Run complete SfM pipeline
        
        Args:
            video_path: Path to input video
            frame_skip: Extract every N-th frame
            max_frames: Maximum number of frames to extract
            base_output_dir: Base directory to save all results
        """
        try:
            image_output_dir = os.path.join(base_output_dir, 'images')
            vis_output_path = os.path.join(base_output_dir, 'reconstruction.png')

            # Step 1: Extract frames (and save to disk)
            self.extract_frames(video_path, frame_skip, max_frames, 
                                output_dir=image_output_dir)
            
            # Step 2: Detect and match features
            self.detect_and_match_features()
            
            if len(self.matches) < 3:
                raise ValueError("Not enough valid image pairs. Try different video.")
            
            # Step 3-4: Initialize reconstruction
            self.initialize_reconstruction()
            
            # Step 5: Incremental reconstruction
            self.incremental_reconstruction()
            
            # Step 5.5: Bundle Adjustment
            if len(self.points_3d) > 0:
                print("\n" + "=" * 60)
                print("STEP 5.5: Bundle Adjustment")
                print("=" * 60)
                self.bundle_adjustment()
                print("Bundle Adjustment complete")
            else:
                print("\nSkipping Bundle Adjustment (no 3D points).")
            
            # Step 6: Visualize
            self.visualize(save_path=vis_output_path)
            
            # Save results
            self.save_results(output_dir=base_output_dir)
            
            print("\n" + "=" * 60)
            print("SfM PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Final statistics:")
            print(f"  - Registered cameras: {len(self.cameras)}/{len(self.images)}")
            print(f"  - 3D points: {len(self.points_3d)}")
            
            if len(self.points_3d) > 0:
                avg_obs = np.mean([len(obs) for obs in self.observations.values()])
                print(f"  - Average observations per point: {avg_obs:.1f}")
            
            print(f"\nFind your results in: {os.path.abspath(base_output_dir)}")
            print(f"Image dataset for COLMAP is in: {os.path.abspath(image_output_dir)}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """
    Main function to run SfM
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Incremental Structure from Motion')
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--frame_skip', type=int, default=10,
                       help='Extract every N-th frame (default: 10)')
    parser.add_argument('--max_frames', type=int, default=30,
                       help='Maximum number of frames to extract (default: 30)')
    parser.add_argument('--focal', type=float, default=800,
                       help='Focal length estimate (default: 800)')
    parser.add_argument('--output', type=str, default='output',
                       help='Base directory to save all results (default: output)')
    
    args = parser.parse_args()
    
    # Create SfM object
    sfm = IncrementalSfM(focal_length=args.focal)
    
    # Run pipeline
    sfm.run(args.video, 
            frame_skip=args.frame_skip, 
            max_frames=args.max_frames,
            base_output_dir=args.output)


if __name__ == '__main__':
    main()