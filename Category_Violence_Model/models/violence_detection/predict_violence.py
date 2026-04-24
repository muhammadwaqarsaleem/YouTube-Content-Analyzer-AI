"""
Violence Detection - Prediction Script for Deployment
Supports multiple input formats: single image, batch images, video files
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import cv2
from pathlib import Path

# Use tf_keras (Keras 2 shim) so .h5 models saved with Keras 2 deserialise
# correctly even when TensorFlow ships Keras 3 (TF >= 2.16).
try:
    import tf_keras as keras_loader
except ImportError:
    import tensorflow.keras as keras_loader

import tensorflow as tf

# ── Keras InputLayer batch_shape compatibility shim ────────────────────────
# Some .h5 models (saved between Keras 2.4-2.8) store `batch_shape` instead
# of `batch_input_shape` in their InputLayer config.  We patch this before
# loading regardless of which Keras backend is in use.
import tensorflow.keras as _tf_keras_native

class _LegacyInputLayer(_tf_keras_native.layers.InputLayer):
    """Accepts old `batch_shape` kwarg and maps it to `batch_input_shape`."""
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config = dict(config)
            config['batch_input_shape'] = config.pop('batch_shape')
        return cls(**config)


_CUSTOM_OBJECTS = {'InputLayer': _LegacyInputLayer}
# ───────────────────────────────────────────────────────────────────────────


class ViolenceDetectionPredictor:
    """
    Predictor class for violence detection model
    Supports inference on images, batches, and videos
    """
    
    def __init__(self, model_path, model_format='h5'):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model
            model_format: Format of the model ('h5', 'savedmodel', 'pkl')
        """
        print(f"Loading model from {model_path}...")
        
        if model_format in ('h5', 'savedmodel'):
            # Use native tensorflow.keras (Keras 3) – the model has Keras-3
            # serialised configs (DTypePolicy etc.), so tf_keras won't work.
            # We still inject _LegacyInputLayer to handle old batch_shape kwarg.
            self.model = _tf_keras_native.models.load_model(
                model_path,
                custom_objects=_CUSTOM_OBJECTS,
                compile=False,
            )
        elif model_format == 'pkl':
            import pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
        
        self.input_shape = (224, 224)
        print("✓ Model loaded successfully!")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Model type: {type(self.model).__name__}")
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for prediction
        
        Args:
            image: Image array (BGR format from OpenCV) or image path
            
        Returns:
            Preprocessed image ready for model prediction
        """
        # If image is a path, load it
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image}")
        
        # Resize to model input size
        img_resized = cv2.resize(image, self.input_shape)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_rgb.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def predict_single(self, image, threshold=0.5, return_confidence=True):
        """
        Predict violence in a single image
        
        Args:
            image: Image array or path
            threshold: Classification threshold
            return_confidence: Whether to return confidence score
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        img_processed = self.preprocess_image(image)
        
        # Make prediction
        prediction_prob = self.model.predict(img_processed, verbose=0)[0][0]
        prediction_class = int(prediction_prob > threshold)
        
        result = {
            'is_violent': bool(prediction_class),
            'confidence': float(prediction_prob),
            'label': 'Violence' if prediction_class else 'Non-Violence',
            'probability_violence': float(prediction_prob),
            'probability_non_violence': float(1 - prediction_prob)
        }
        
        return result
    
    def predict_batch(self, image_paths, threshold=0.5, batch_size=32):
        """
        Predict violence in multiple images
        
        Args:
            image_paths: List of image paths
            threshold: Classification threshold
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction dictionaries
        """
        print(f"Processing {len(image_paths)} images...")
        
        # Load and preprocess all images
        images_processed = []
        for img_path in image_paths:
            try:
                img_processed = self.preprocess_image(img_path)
                images_processed.append(img_processed)
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
        
        if len(images_processed) == 0:
            return []
        
        # Stack into batch
        images_batch = np.vstack(images_processed)
        
        # Make predictions
        predictions_prob = self.model.predict(images_batch, verbose=0).flatten()
        predictions_class = (predictions_prob > threshold).astype(int)
        
        results = []
        for i, (prob, pred_class) in enumerate(zip(predictions_prob, predictions_class)):
            result = {
                'image_path': image_paths[i],
                'is_violent': bool(pred_class),
                'confidence': float(prob),
                'label': 'Violence' if pred_class else 'Non-Violence',
                'probability_violence': float(prob),
                'probability_non_violence': float(1 - prob)
            }
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)} images")
        
        return results
    
    def predict_video(self, video_path, output_path=None, threshold=0.5, 
                     frame_interval=1, show_progress=True):
        """
        Detect violence in video frames
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save output video with annotations
            threshold: Classification threshold
            frame_interval: Process every Nth frame
            show_progress: Show progress bar
            
        Returns:
            Dictionary with video analysis results
        """
        print(f"Analyzing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Processing interval: Every {frame_interval} frame(s)")
        
        # Prepare output video writer if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        violent_frames = []
        non_violent_frames = []
        predictions_timeline = []
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames based on interval
            if frame_count % frame_interval != 0:
                if output_path:
                    out.write(frame)
                continue
            
            # Preprocess frame
            img_resized = cv2.resize(frame, self.input_shape)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict
            prediction_prob = self.model.predict(img_batch, verbose=0)[0][0]
            prediction_class = int(prediction_prob > threshold)
            
            # Store results
            result = {
                'frame_number': frame_count,
                'timestamp_seconds': frame_count / fps,
                'is_violent': bool(prediction_class),
                'confidence': float(prediction_prob),
                'label': 'Violence' if prediction_class else 'Non-Violence'
            }
            predictions_timeline.append(result)
            
            if prediction_class:
                violent_frames.append(frame_count)
            else:
                non_violent_frames.append(frame_count)
            
            # Annotate frame if saving output
            if output_path:
                label = f"VIOLENCE: {prediction_prob:.2f}" if prediction_class else f"SAFE: {1-prediction_prob:.2f}"
                color = (0, 0, 255) if prediction_class else (0, 255, 0)
                
                cv2.putText(frame, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                out.write(frame)
            
            # Show progress
            if show_progress and frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        # Calculate statistics
        total_processed = len(violent_frames) + len(non_violent_frames)
        violence_percentage = (len(violent_frames) / total_processed * 100) if total_processed > 0 else 0
        
        results = {
            'video_path': video_path,
            'output_video_path': output_path,
            'total_frames': total_frames,
            'frames_processed': total_processed,
            'violent_frames_count': len(violent_frames),
            'non_violent_frames_count': len(non_violent_frames),
            'violence_percentage': violence_percentage,
            'violent_frame_numbers': violent_frames,
            'predictions_timeline': predictions_timeline,
            'overall_assessment': 'VIOLENT' if len(violent_frames) > 0 else 'NON-VIOLENT'
        }
        
        print(f"\n✓ Video analysis complete!")
        print(f"  Frames processed: {total_processed}")
        print(f"  Violent frames: {len(violent_frames)} ({violence_percentage:.2f}%)")
        print(f"  Non-violent frames: {len(non_violent_frames)}")
        if output_path:
            print(f"  Output video saved: {output_path}")
        
        return results
    
    def predict_from_directory(self, directory_path, threshold=0.5, 
                               extensions=('.jpg', '.jpeg', '.png')):
        """
        Predict violence for all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            threshold: Classification threshold
            extensions: Valid image extensions
            
        Returns:
            Dictionary with predictions and statistics
        """
        print(f"Scanning directory: {directory_path}")
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(list(Path(directory_path).glob(f'*{ext}')))
            image_files.extend(list(Path(directory_path).glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            return {'error': 'No images found'}
        
        # Make predictions
        image_paths = [str(p) for p in image_files]
        predictions = self.predict_batch(image_paths, threshold=threshold)
        
        # Calculate statistics
        violent_count = sum(1 for p in predictions if p['is_violent'])
        non_violent_count = len(predictions) - violent_count
        
        results = {
            'directory_path': directory_path,
            'total_images': len(predictions),
            'violent_images': violent_count,
            'non_violent_images': non_violent_count,
            'violence_percentage': (violent_count / len(predictions) * 100) if len(predictions) > 0 else 0,
            'predictions': predictions
        }
        
        print(f"\n✓ Directory analysis complete!")
        print(f"  Total images: {len(predictions)}")
        print(f"  Violent: {violent_count} ({results['violence_percentage']:.2f}%)")
        print(f"  Non-violent: {non_violent_count}")
        
        return results


def main():
    """
    Example usage of the violence detection predictor
    """
    print("="*70)
    print("VIOLENCE DETECTION - PREDICTION DEMO")
    print("="*70 + "\n")
    
    # Load the model
    model_path = 'violence_detection_model_resnet.h5'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using main_violence_detection.py")
        return
    
    predictor = ViolenceDetectionPredictor(model_path, model_format='h5')
    
    # Example 1: Single image prediction
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*70)
    
    test_image = 'test_image.jpg'  # Replace with actual image path
    if os.path.exists(test_image):
        result = predictor.predict_single(test_image)
        print(f"\nPrediction for {test_image}:")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probability of Violence: {result['probability_violence']*100:.2f}%")
    else:
        print(f"Test image not found: {test_image}")
        print("Skipping single image prediction")
    
    # Example 2: Batch prediction
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Prediction from Directory")
    print("="*70)
    
    test_dir = 'violence_frames/violence'  # Replace with actual directory
    if os.path.exists(test_dir):
        results = predictor.predict_from_directory(test_dir, threshold=0.5)
        if 'error' not in results:
            print(f"\nDirectory Analysis Results:")
            print(f"  Total images: {results['total_images']}")
            print(f"  Violent: {results['violent_images']}")
            print(f"  Non-violent: {results['non_violent_images']}")
    else:
        print(f"Test directory not found: {test_dir}")
        print("Skipping batch prediction")
    
    # Example 3: Video prediction
    print("\n" + "="*70)
    print("EXAMPLE 3: Video Violence Detection")
    print("="*70)
    
    test_video = 'test_video.mp4'  # Replace with actual video path
    if os.path.exists(test_video):
        results = predictor.predict_video(
            test_video,
            output_path='output_annotated.mp4',
            threshold=0.5,
            frame_interval=5  # Process every 5th frame for speed
        )
        print(f"\nVideo Analysis Results:")
        print(f"  Overall Assessment: {results['overall_assessment']}")
        print(f"  Violent Frames: {results['violent_frames_count']}")
        print(f"  Violence Percentage: {results['violence_percentage']:.2f}%")
    else:
        print(f"Test video not found: {test_video}")
        print("Skipping video prediction")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
