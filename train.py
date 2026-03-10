import cv2
import numpy as np
import pickle
import os
from tkinter import Tk, filedialog, simpledialog

class FaceTrainer:
    def __init__(self):
        self.encodings = []
        self.names = []
        self.model_file = 'trained_model.pkl'
        
        # Load existing model if exists
        self.load_model()
    
    def extract_features(self, face_img):
        """Extract 128-dimension features from face image to match app.py"""
        try:
            # Resize face to standard size
            face = cv2.resize(face_img, (128, 128))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            features = []
            
            # 1. HOG-like features (64 dimensions)
            # Calculate gradients
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
            mag, ang = cv2.cartToPolar(gx, gy)
            
            # Create 8x8 cells, 8 orientations = 64 features
            cell_size = 16
            for i in range(0, 128, cell_size):
                for j in range(0, 128, cell_size):
                    cell_mag = mag[i:i+cell_size, j:j+cell_size]
                    hist = np.histogram(ang[i:i+cell_size, j:j+cell_size], 
                                       bins=8, weights=cell_mag, range=(0, 2*np.pi))[0]
                    features.extend(hist)
            
            # 2. LBP features (32 dimensions)
            lbp = np.zeros_like(gray)
            for i in range(1, gray.shape[0]-1):
                for j in range(1, gray.shape[1]-1):
                    center = gray[i, j]
                    code = 0
                    code |= (gray[i-1, j-1] > center) << 7
                    code |= (gray[i-1, j] > center) << 6
                    code |= (gray[i-1, j+1] > center) << 5
                    code |= (gray[i, j+1] > center) << 4
                    code |= (gray[i+1, j+1] > center) << 3
                    code |= (gray[i+1, j] > center) << 2
                    code |= (gray[i+1, j-1] > center) << 1
                    code |= (gray[i, j-1] > center) << 0
                    lbp[i, j] = code
            
            # LBP histogram (32 bins)
            lbp_hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [32], [0, 256])
            lbp_hist = lbp_hist.flatten() / (lbp_hist.sum() + 1e-6)
            features.extend(lbp_hist)
            
            # 3. Color and texture features (32 dimensions)
            # Mean and std of blocks
            for i in range(0, 128, 32):
                for j in range(0, 128, 32):
                    block = gray[i:i+32, j:j+32]
                    features.append(np.mean(block) / 255.0)
                    features.append(np.std(block) / 255.0)
            
            # Make sure we have exactly 128 features
            features = np.array(features[:128], dtype=np.float32)
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-6)
            
            print(f"  ✓ Features extracted: {len(features)} dimensions")
            return features
            
        except Exception as e:
            print(f"  ✗ Error extracting features: {e}")
            return None
    
    def add_member(self):
        """Add a new member"""
        root = Tk()
        root.withdraw()
        
        name = simpledialog.askstring("Member Name", "Enter member name:")
        if not name:
            return
        
        files = filedialog.askopenfilenames(
            title=f"Select images for {name}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not files:
            return
        
        print(f"\n📸 Processing {len(files)} images for {name}...")
        print("-" * 40)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        success = 0
        for i, file in enumerate(files):
            print(f"  Image {i+1}/{len(files)}: {os.path.basename(file)}")
            
            # Read image
            img = cv2.imread(file)
            if img is None:
                print(f"    ✗ Cannot read image")
                continue
            
            # Detect face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                print(f"    ✗ No face found")
                continue
            
            # Use first face
            x,y,w,h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            
            # Extract features
            features = self.extract_features(face_roi)
            
            if features is not None:
                self.encodings.append(features)
                self.names.append(name)
                success += 1
                print(f"    ✓ Face encoded successfully")
            else:
                print(f"    ✗ Feature extraction failed")
        
        print("-" * 40)
        print(f"✅ Added {success}/{len(files)} faces for {name}")
        self.save_model()
    
    def save_model(self):
        """Save model to file"""
        data = {
            'encodings': self.encodings,
            'names': self.names
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n💾 Model saved to {self.model_file}")
        print(f"📊 Total faces: {len(self.encodings)}")
        if self.encodings:
            print(f"📐 Feature dimension: {len(self.encodings[0])}")
    
    def load_model(self):
        """Load existing model"""
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                data = pickle.load(f)
            self.encodings = data['encodings']
            self.names = data['names']
            print(f"📂 Loaded existing model with {len(self.encodings)} faces")
            if self.encodings:
                print(f"📐 Feature dimension: {len(self.encodings[0])}")
    
    def show_members(self):
        """Show all members"""
        if not self.names:
            print("📭 No members yet")
            return
        
        unique = set(self.names)
        print("\n📋 Registered Members:")
        for name in unique:
            count = self.names.count(name)
            print(f"  • {name}: {count} face(s)")
        
        if self.encodings:
            print(f"\n📐 Feature dimension: {len(self.encodings[0])}")

def main():
    trainer = FaceTrainer()
    
    while True:
        print("\n" + "="*50)
        print("🤖 FACE RECOGNITION TRAINER")
        print("="*50)
        print("1. Add Member")
        print("2. View Members")
        print("3. Exit")
        print("="*50)
        
        choice = input("Choice (1-3): ")
        
        if choice == '1':
            trainer.add_member()
        elif choice == '2':
            trainer.show_members()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()