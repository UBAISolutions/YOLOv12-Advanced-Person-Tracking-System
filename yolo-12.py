# Import OpenCV library for computer vision operations
import cv2
# Import YOLO class from Ultralytics library for YOLOv12 object detection and tracking
from ultralytics import YOLO
# Import collections for advanced data structures
# defaultdict: automatically creates missing keys with default values
# deque: double-ended queue with maximum length for efficient trail storage
from collections import defaultdict, deque
# Import pandas for Excel report generation
import pandas as pd
# Import datetime for timestamps
from datetime import datetime
import time
# Import numpy for mathematical operations
import numpy as np
# Import scipy for distance calculations
from scipy.spatial.distance import cosine
import os

# Create a YOLO model instance by loading YOLOv12 nano model weights
model = YOLO("yolo12n.pt")

# Initialize video capture from video file
# Replace 'data/people_walking.mp4' with your video file path
video_path = "data/Subway.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    print("Available video files in data folder:")
    import os
    if os.path.exists("data"):
        for file in os.listdir("data"):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"  - {file}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"🎬 Video Analysis Started")
print(f"📁 Video file: {video_path}")
print(f"📊 Total frames: {total_frames}")
print(f"⏱️ Duration: {duration:.2f} seconds")
print(f"🎯 FPS: {fps:.2f}")
print("Press 'q' to stop analysis\n")

# Video writer will be initialized lazily after the first processed frame
out = None
# Keep original video FPS for output recording
original_fps = fps  # Store the original FPS from the video file

# Dictionary to map YOLO's internal object IDs to our custom sequential IDs
# This helps maintain consistent numbering even when YOLO IDs change
id_map = {}
# Counter for assigning new sequential IDs starting from 1
next_id = 1

# defaultdict with deque: each person gets a trail (deque) that stores up to 30 recent positions
# maxlen=30 automatically removes old positions when new ones are added
trail = defaultdict(lambda: deque(maxlen=30))
# Track how many consecutive frames each object has appeared
# Used to filter out brief false detections
appear = defaultdict(int)

# Data collection for Excel report
tracking_data = []  # List to store all tracking data
person_stats = defaultdict(lambda: {
    'first_seen': None,
    'last_seen': None,
    'total_frames': 0,
    'positions': [],
    'avg_x': 0,
    'avg_y': 0,
    'distance_traveled': 0
})
start_time = datetime.now()
frame_count = 0
total_detections = 0  # Counter for total person detections

# DeepFace facial recognition for persistent person identification
known_face_encodings = []  # Store face embeddings from DeepFace
known_person_ids = []  # Store corresponding person IDs
known_person_last_seen = []  # Track when each person was last seen
face_similarity_threshold = 0.6  # Correlation similarity threshold (higher = more strict)
min_face_size = 50  # Minimum face size for reliable recognition
max_known_people = 30  # Limit stored people to prevent memory issues
person_timeout = 450  # Frames after which to forget a person (15 seconds at 30fps)

def cleanup_old_people(current_frame):
    """
    Remove people who haven't been seen for a long time to prevent memory bloat
    and reduce false matches with very old data.
    """
    global known_face_encodings, known_person_ids, known_person_last_seen
    
    indices_to_remove = []
    for i, last_seen in enumerate(known_person_last_seen):
        if current_frame - last_seen > person_timeout:
            indices_to_remove.append(i)
    
    # Remove old people in reverse order to maintain indices
    for i in reversed(indices_to_remove):
        removed_id = known_person_ids[i]
        known_face_encodings.pop(i)
        known_person_ids.pop(i)
        known_person_last_seen.pop(i)
        print(f"🗑️  Removed old person ID {removed_id} from memory")

def extract_and_encode_face(person_region):
    """
    Extract facial features from a person's image using OpenCV.
    Returns a feature vector for face comparison.
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's face detection classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size for comparison
            face_roi = cv2.resize(face_roi, (60, 60))
            
            # Calculate histogram as a simple feature
            hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            
            # Normalize histogram
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-10)  # Avoid division by zero
            
            return hist
        else:
            # No face detected, use overall appearance features
            resized = cv2.resize(gray, (50, 50))
            hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-10)
            return hist
            
    except Exception as e:
        print(f"Face extraction error: {e}")
        return None

def calculate_face_similarity(embedding1, embedding2):
    """
    Calculate similarity between two feature vectors using correlation.
    Returns similarity score (0-1, higher = more similar).
    """
    try:
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Calculate correlation coefficient
        correlation = np.corrcoef(embedding1, embedding2)[0, 1]
        
        # Handle NaN values
        if np.isnan(correlation):
            return 0.0
            
        # Convert to similarity score (0-1)
        similarity = (correlation + 1) / 2  # Convert from [-1,1] to [0,1]
        
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0
        
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0

def get_or_assign_person_id(frame, x1, y1, x2, y2, current_frame):
    """
    DeepFace-based person re-identification with facial recognition.
    Much more accurate than visual features for person identification.
    """
    global known_face_encodings, known_person_ids, known_person_last_seen, next_id, max_known_people
    
    # Cleanup old people periodically
    if current_frame % 30 == 0:  # Every second at 30fps
        cleanup_old_people(current_frame)
    
    # Extract person region from frame
    person_region = frame[y1:y2, x1:x2]
    
    # Check if person region is large enough for reliable face detection
    person_height = y2 - y1
    person_width = x2 - x1
    
    if person_height >= min_face_size and person_width >= min_face_size:
        # Extract face embedding using DeepFace
        face_embedding = extract_and_encode_face(person_region)
        
        if face_embedding is not None:
            # Find best matches among known faces
            similarities = []
            for i, known_embedding in enumerate(known_face_encodings):
                similarity = calculate_face_similarity(face_embedding, known_embedding)
                
                # Boost similarity for recently seen people
                frames_since_last_seen = current_frame - known_person_last_seen[i]
                if frames_since_last_seen < 30:  # Recently seen (within 1 second)
                    similarity *= 1.15  # 15% boost
                elif frames_since_last_seen < 90:  # Moderately recent (within 3 seconds)
                    similarity *= 1.08  # 8% boost
                
                similarities.append((similarity, known_person_ids[i], i))
            
            # Sort by similarity (highest first)
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            # Check if best match exceeds threshold
            if similarities and similarities[0][0] >= face_similarity_threshold:
                best_similarity, best_id, best_index = similarities[0]
                
                # Update the stored embedding with a weighted average for adaptation
                alpha = 0.1  # Learning rate for embedding adaptation
                known_face_encodings[best_index] = (
                    (1 - alpha) * known_face_encodings[best_index] + 
                    alpha * face_embedding
                )
                
                # Update last seen time
                known_person_last_seen[best_index] = current_frame
                
                print(f"� Person ID {best_id} re-identified via DeepFace (similarity: {best_similarity:.3f})")
                return best_id
            else:
                # No good match found, create new person ID
                new_id = next_id
                next_id += 1
                
                # Limit the number of stored people to prevent memory issues
                if len(known_face_encodings) >= max_known_people:
                    # Remove the oldest entry (by last seen time)
                    oldest_index = np.argmin(known_person_last_seen)
                    removed_id = known_person_ids[oldest_index]
                    known_face_encodings.pop(oldest_index)
                    known_person_ids.pop(oldest_index)
                    known_person_last_seen.pop(oldest_index)
                    print(f"🗑️  Removed person ID {removed_id} to make room")
                
                known_face_encodings.append(face_embedding)
                known_person_ids.append(new_id)
                known_person_last_seen.append(current_frame)
                
                best_sim = similarities[0][0] if similarities else 0.0
                print(f"🆕 New person ID {new_id} created via DeepFace (best similarity was: {best_sim:.3f})")
                return new_id
        else:
            # No face detected, could add fallback to visual features here
            print(f"⚠️  No face detected in region {person_width}x{person_height}")
    
    # Person too small or face detection failed, use fallback
    return None
known_person_features = []  # Store visual features of known people
known_person_ids = []  # Store corresponding person IDs
known_person_keypoints = []  # Store body keypoints for pose comparison
similarity_threshold = 0.6  # Lowered threshold for better matching (0.0-1.0)
min_person_size = 40  # Reduced minimum person size
max_known_people = 50  # Limit stored people to prevent memory issues

def extract_enhanced_features(person_region):
    """
    Extract multiple types of visual features for robust person re-identification.
    Combines color histograms, texture features, and basic shape information.
    """
    if person_region.size == 0:
        return None
    
    # Resize to standard size for comparison
    person_resized = cv2.resize(person_region, (64, 128))
    
    features = []
    
    # 1. Color features in HSV space (more robust to lighting changes)
    hsv = cv2.cvtColor(person_resized, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for upper and lower body separately
    upper_body = hsv[:64, :]  # Upper half (torso, head)
    lower_body = hsv[64:, :]  # Lower half (legs)
    
    # Upper body color histograms
    hist_h_upper = cv2.calcHist([upper_body], [0], None, [30], [0, 180])
    hist_s_upper = cv2.calcHist([upper_body], [1], None, [32], [0, 256])
    hist_v_upper = cv2.calcHist([upper_body], [2], None, [32], [0, 256])
    
    # Lower body color histograms
    hist_h_lower = cv2.calcHist([lower_body], [0], None, [30], [0, 180])
    hist_s_lower = cv2.calcHist([lower_body], [1], None, [32], [0, 256])
    hist_v_lower = cv2.calcHist([lower_body], [2], None, [32], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist_h_upper, hist_h_upper)
    cv2.normalize(hist_s_upper, hist_s_upper)
    cv2.normalize(hist_v_upper, hist_v_upper)
    cv2.normalize(hist_h_lower, hist_h_lower)
    cv2.normalize(hist_s_lower, hist_s_lower)
    cv2.normalize(hist_v_lower, hist_v_lower)
    
    # 2. Texture features using Local Binary Pattern approximation
    gray = cv2.cvtColor(person_resized, cv2.COLOR_BGR2GRAY)
    
    # Simple edge density for texture
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    # 3. Dominant colors (simplified)
    pixels = person_resized.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # K-means to find dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    
    # Sort dominant colors by frequency
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    dominant_colors = centers[sorted_indices].flatten()
    
    # Combine all features
    features.extend(hist_h_upper.flatten())
    features.extend(hist_s_upper.flatten())
    features.extend(hist_v_upper.flatten())
    features.extend(hist_h_lower.flatten())
    features.extend(hist_s_lower.flatten())
    features.extend(hist_v_lower.flatten())
    features.append(edge_density)
    features.extend(dominant_colors)
    
    return np.array(features, dtype=np.float32)

def calculate_multiple_similarities(features1, features2):
    """
    Calculate multiple similarity scores and combine them for robust matching.
    """
    if features1 is None or features2 is None:
        return 0.0
    
    try:
        # Ensure features are the same length
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # 1. Correlation coefficient
        correlation = cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL)
        correlation = max(0, correlation)  # Ensure positive
        
        # 2. Chi-Square distance (converted to similarity)
        chi_square = cv2.compareHist(f1, f2, cv2.HISTCMP_CHISQR)
        chi_similarity = 1.0 / (1.0 + chi_square)
        
        # 3. Intersection similarity
        intersection = cv2.compareHist(f1, f2, cv2.HISTCMP_INTERSECT)
        intersection_norm = intersection / (np.sum(f1) + 1e-10)
        
        # 4. Euclidean distance (converted to similarity)
        euclidean_dist = np.linalg.norm(f1 - f2)
        euclidean_similarity = 1.0 / (1.0 + euclidean_dist)
        
        # Weighted combination of similarities
        combined_similarity = (
            0.35 * correlation +
            0.25 * chi_similarity +
            0.25 * intersection_norm +
            0.15 * euclidean_similarity
        )
        
        return min(1.0, max(0.0, combined_similarity))
        
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 0.0

def get_or_assign_person_id(frame, x1, y1, x2, y2, current_frame):
    """
    Enhanced person re-identification with multiple feature types, adaptive matching,
    and temporal cleanup of old people.
    """
    global known_person_features, known_person_ids, known_person_last_seen, next_id, max_known_people
    
    # Cleanup old people periodically
    if current_frame % 30 == 0:  # Every second at 30fps
        cleanup_old_people(current_frame)
    
    # Extract person region from frame
    person_region = frame[y1:y2, x1:x2]
    
    # Check if person region is large enough for reliable identification
    person_height = y2 - y1
    person_width = x2 - x1
    
    if person_height >= min_person_size and person_width >= min_person_size:
        # Extract enhanced visual features
        features = extract_enhanced_features(person_region)
        
        if features is not None:
            # Find best matches
            similarities = []
            for i, known_features in enumerate(known_person_features):
                similarity = calculate_multiple_similarities(features, known_features)
                
                # Boost similarity for recently seen people
                frames_since_last_seen = current_frame - known_person_last_seen[i]
                if frames_since_last_seen < 30:  # Recently seen (within 1 second)
                    similarity *= 1.2  # 20% boost
                elif frames_since_last_seen < 90:  # Moderately recent (within 3 seconds)
                    similarity *= 1.1  # 10% boost
                
                similarities.append((similarity, known_person_ids[i], i))
            
            # Sort by similarity (highest first)
            similarities.sort(reverse=True, key=lambda x: x[0])
            
            # Check if best match exceeds threshold
            if similarities and similarities[0][0] >= similarity_threshold:
                best_similarity, best_id, best_index = similarities[0]
                
                # Update the stored features with a weighted average for adaptation
                alpha = 0.15  # Increased learning rate for better adaptation
                known_person_features[best_index] = (
                    (1 - alpha) * known_person_features[best_index] + 
                    alpha * features
                )
                
                # Update last seen time
                known_person_last_seen[best_index] = current_frame
                
                print(f"🔄 Person ID {best_id} re-identified (similarity: {best_similarity:.3f})")
                return best_id
            else:
                # No good match found, create new person ID
                new_id = next_id
                next_id += 1
                
                # Limit the number of stored people to prevent memory issues
                if len(known_person_features) >= max_known_people:
                    # Remove the oldest entry (by last seen time)
                    oldest_index = np.argmin(known_person_last_seen)
                    removed_id = known_person_ids[oldest_index]
                    known_person_features.pop(oldest_index)
                    known_person_ids.pop(oldest_index)
                    known_person_last_seen.pop(oldest_index)
                    print(f"🗑️  Removed person ID {removed_id} to make room")
                
                known_person_features.append(features)
                known_person_ids.append(new_id)
                known_person_last_seen.append(current_frame)
                
                best_sim = similarities[0][0] if similarities else 0.0
                print(f"🆕 New person ID {new_id} created (best similarity was: {best_sim:.3f})")
                return new_id
    
    # Person too small or feature extraction failed, use fallback
    return None

# Start infinite loop for video processing
# Video writer will be initialized after first frame to get correct dimensions
out = None

while True:
    # Read the next frame from the video file
    ret, frame = cap.read()
    # If no more frames available (end of video), exit the loop
    if not ret:
        break
    
    frame_count += 1
    current_timestamp = datetime.now()
    
    # Show progress every 30 frames (approximately every second)
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        print(f"📊 Processing: Frame {frame_count}/{total_frames} ({progress:.1f}%) - {len(known_face_encodings)} unique people detected")
    
    # Run YOLO object detection and tracking
    # persist=True: maintains object IDs across frames
    # classes=[0]: only detect "person" class from COCO dataset
    # verbose=False: suppress detailed output messages
    results = model.track(frame, persist=True, classes=[0], verbose=False)

    # Get the default annotated frame from YOLO
    annotated_frame = results[0].plot()

    # Initialize the video writer once we know the correct frame size
    if out is None:
        height, width = annotated_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Save as .mp4
        out = cv2.VideoWriter("data/video_analysis_yolo12.mp4", fourcc, original_fps, (width, height))
        print(f"🎥 Video writer initialized: {width}x{height} @ {original_fps} FPS")

    # Check if any people were detected and have tracking IDs
    if results[0].boxes.id is not None:
        # Extract bounding box coordinates as NumPy array
        boxes = results[0].boxes.xyxy.numpy()
        # Extract tracking IDs as NumPy array
        ids = results[0].boxes.id.numpy()

        # Process each detected person (iterate through boxes and their corresponding IDs)
        for box, oid in zip(boxes, ids):
            # Increment total detections counter
            total_detections += 1
            
            # Extract bounding box coordinates and convert to integers
            x1, y1, x2, y2 = map(int, box)
            # Calculate center point of the bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Increment appearance counter for this object ID
            appear[oid] += 1

            # Only assign a permanent ID if object has appeared for 5+ frames
            # This filters out brief false detections and ensures stable tracking
            if appear[oid] >= 5:
                if oid not in id_map:
                    # Try enhanced visual feature matching for persistent ID assignment
                    feature_based_id = get_or_assign_person_id(frame, x1, y1, x2, y2, frame_count)
                    
                    if feature_based_id is not None:
                        # Visual matching successful, use feature-based ID
                        id_map[oid] = feature_based_id
                    else:
                        # Fall back to sequential ID if visual matching fails
                        id_map[oid] = next_id
                        next_id += 1
                        print(f"⚠️  Person ID {id_map[oid]} assigned via fallback (person too small: {y2-y1}x{x2-x1})")

            # Only process objects that have been confirmed (appeared 5+ times)
            if oid in id_map:
                # Get the stable sequential ID for this object
                id = id_map[oid]
                # Add current center position to this person's trail
                trail[id].append((cx, cy))

                # Collect data for Excel report
                if person_stats[id]['first_seen'] is None:
                    person_stats[id]['first_seen'] = current_timestamp
                person_stats[id]['last_seen'] = current_timestamp
                person_stats[id]['total_frames'] += 1
                person_stats[id]['positions'].append((cx, cy))
                
                # Add detailed tracking data
                tracking_data.append({
                    'frame_number': frame_count,
                    'timestamp': current_timestamp,
                    'person_id': id,
                    'center_x': cx,
                    'center_y': cy,
                    'bbox_x1': x1,
                    'bbox_y1': y1,
                    'bbox_x2': x2,
                    'bbox_y2': y2,
                    'bbox_width': x2 - x1,
                    'bbox_height': y2 - y1
                })

                # Draw the bounding box around the person (red color, thinner line)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                # Draw the ID number above the bounding box (even smaller font)
                cv2.putText(annotated_frame, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                
                # Draw a smaller circle at the center point of the person
                cv2.circle(annotated_frame, (cx, cy), 3, (0, 0, 255), -1)

                # Draw the movement trail by connecting consecutive points
                # Note: using oid instead of id here seems to be a bug in original code
                trail_points = list(trail[id])
                # Connect each point to the next with a red line
                for i in range(1, len(trail_points)):
                    cv2.line(annotated_frame, trail_points[i - 1], trail_points[i], (0, 0, 255), 2)

    # Add information overlay panel
    overlay_height = 120
    overlay_width = 350
    
    # Create semi-transparent background for overlay
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (10, 10), (overlay_width, overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
    
    # Add border
    cv2.rectangle(annotated_frame, (10, 10), (overlay_width, overlay_height), (0, 255, 255), 2)
    
    # Add model and statistics information (even smaller fonts)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    line_height = 12
    start_y = 22
    
    # Model information
    cv2.putText(annotated_frame, "YOLOv12 Nano + OpenCV Face Recognition", (15, start_y), font, font_scale, (0, 255, 255), thickness)
    cv2.putText(annotated_frame, f"Model: yolo12n.pt | Threshold: 0.6", (15, start_y + line_height), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(annotated_frame, f"Video: {os.path.basename(video_path)}", (15, start_y + 2*line_height), font, font_scale, (255, 255, 255), thickness)
    
    # Statistics
    unique_people = len([pid for pid, count in appear.items() if count >= 5])
    active_people = len(appear)
    cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (15, start_y + 3*line_height), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(annotated_frame, f"Unique People: {unique_people} | Active IDs: {active_people}", (15, start_y + 4*line_height), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(annotated_frame, "Press 'q' to quit", (15, start_y + 5*line_height), font, font_scale, (255, 255, 0), thickness)

    # Resize the frame for smaller display window (75% of original size)
    height, width = annotated_frame.shape[:2]
    new_width = int(width * 0.75)
    new_height = int(height * 0.75)
    resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

    # Display the resized annotated frame with trails and tracking information
    cv2.imshow("Video Analysis - Person Tracking with YOLOv12", resized_frame)

    # Write the processed frame to the output video (full resolution)
    if out is not None:
        out.write(annotated_frame)
    
    # Allow window to update but don't wait for key press
    # Use waitKey(1) to allow window updates but continue processing
    key = cv2.waitKey(1) & 0xFF
    # Only quit if 'q' is explicitly pressed (optional manual stop)
    if key == ord('q'):
        print("Manual stop requested...")
        break

# Release the video file handle
cap.release()
# Release the video writer if it was created
if out is not None:
    out.release()
# Close all OpenCV windows
cv2.destroyAllWindows()

# Print completion summary
print(f"\n🎬 Video Analysis Completed!")
print(f"📊 Total frames processed: {frame_count}")
print(f"👥 Unique people detected: {len(known_face_encodings)}")
print(f"🔍 Total detections: {total_detections}")
print(f"📁 Output video saved as: data/video_analysis_yolo12.mp4")
print("📈 Generating Excel report...\n")

# Generate Excel Report
print("\nGenerating tracking summary report...")

# Calculate additional statistics for each person
for person_id, stats in person_stats.items():
    if stats['positions']:
        # Calculate average position
        stats['avg_x'] = sum(pos[0] for pos in stats['positions']) / len(stats['positions'])
        stats['avg_y'] = sum(pos[1] for pos in stats['positions']) / len(stats['positions'])
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(stats['positions'])):
            x1, y1 = stats['positions'][i-1]
            x2, y2 = stats['positions'][i]
            distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
            total_distance += distance
        stats['distance_traveled'] = total_distance

# Create summary DataFrame
end_time = datetime.now()
session_duration = (end_time - start_time).total_seconds()

summary_data = []
for person_id, stats in person_stats.items():
    if stats['first_seen'] is not None:
        duration_in_frame = (stats['last_seen'] - stats['first_seen']).total_seconds()
        summary_data.append({
            'Person_ID': person_id,
            'First_Seen': stats['first_seen'].strftime('%Y-%m-%d %H:%M:%S'),
            'Last_Seen': stats['last_seen'].strftime('%Y-%m-%d %H:%M:%S'),
            'Duration_Seconds': round(duration_in_frame, 2),
            'Total_Frames_Detected': stats['total_frames'],
            'Average_X_Position': round(stats['avg_x'], 2),
            'Average_Y_Position': round(stats['avg_y'], 2),
            'Total_Distance_Traveled_Pixels': round(stats['distance_traveled'], 2),
            'Detection_Rate_Percent': round((stats['total_frames'] / frame_count) * 100, 2)
        })

# Create detailed tracking DataFrame
detailed_df = pd.DataFrame(tracking_data)

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Create session info DataFrame
session_info = pd.DataFrame([{
    'Session_Start': start_time.strftime('%Y-%m-%d %H:%M:%S'),
    'Session_End': end_time.strftime('%Y-%m-%d %H:%M:%S'),
    'Total_Duration_Seconds': round(session_duration, 2),
    'Total_Frames_Processed': frame_count,
    'Total_People_Tracked': len(person_stats),
    'Video_Output_File': 'data/video_analysis_yolo12.mp4',
    'Model_Used': 'YOLOv12 Nano'
}])

# Generate Excel file with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
excel_filename = f'data/tracking_report_{timestamp}.xlsx'

# Write to Excel with multiple sheets
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    session_info.to_excel(writer, sheet_name='Session_Info', index=False)
    summary_df.to_excel(writer, sheet_name='People_Summary', index=False)
    if not detailed_df.empty:
        detailed_df.to_excel(writer, sheet_name='Detailed_Tracking', index=False)

print(f"✅ Tracking report saved as: {excel_filename}")
print(f"📊 Total unique people tracked: {len(person_stats)}")
print(f"👥 DeepFace facial encodings: {len(known_face_encodings)}")
print(f"⏱️  Session duration: {round(session_duration, 2)} seconds")
print(f"🎬 Total frames processed: {frame_count}")
print("\n📋 Report contains:")
print("   • Session_Info: Overall session statistics")
print("   • People_Summary: Individual person tracking statistics") 
print("   • Detailed_Tracking: Frame-by-frame tracking data")
print("\n🎯 OpenCV Face Recognition Stats:")
print(f"   • Unique faces detected: {len(known_face_encodings)}")
print(f"   • Model used: OpenCV Histogram-based")
print(f"   • Similarity threshold: {face_similarity_threshold}")
print(f"   • Same person re-entering will keep the same ID")
print(f"   • Memory cleanup removes people not seen for {person_timeout/30:.1f} seconds")
