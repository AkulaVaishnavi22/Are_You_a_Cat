import os
from PIL import Image

DATA_DIR = "data"
CLASS_NAMES = ['cats', 'dogs', 'humans']

print("🔍 Starting Automated Data Integrity Scan...")

total_images_scanned = 0
corrupt_files_found = 0
bad_dimensions_found = 0

# Track file sizes to catch exact duplicates
seen_file_sizes = {}
duplicate_count = 0

for folder in CLASS_NAMES:
    folder_path = os.path.join(DATA_DIR, folder)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder '{folder_path}' does not exist yet.")
        continue
        
    print(f"\n📂 Scanning sub-directory: `{folder_path}`...")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip hidden operating system metadata files if they exist
        if os.path.isdir(file_path) or filename.startswith('.'):
            continue
            
        total_images_scanned += 1
        
        # Guardrail 1: Check File Extension Integrity
        valid_extensions = ('.jpg', '.jpeg', '.png')
        if not filename.lower().endswith(valid_extensions):
            print(f"❌ [INVALID EXTENSION] File '{filename}' is not a valid format type.")
            corrupt_files_found += 1
            continue
            
        try:
            # Guardrail 2: Test Bitstream & Decoding Corruption
            with Image.open(file_path) as img:
                img.verify() # Decodes image byte integrity quickly
                
            # Re-open to inspect spatial dimensions safely
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Guardrail 3: Track Extreme Aspect Ratio Outliers (Squish Prevention)
                aspect_ratio = width / height
                if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                    print(f"⚠️ [ASPECT RATIO WARNING] '{filename}' is heavily distorted ({width}x{height}).")
                    bad_dimensions_found += 1
                
                # Guardrail 4: Duplicate Content Detection via File Fingerprinting
                file_size = os.path.getsize(file_path)
                if file_size in seen_file_sizes:
                    print(f"👥 [DUPLICATE DETECTED] '{filename}' matches original file size of '{seen_file_sizes[file_size]}'.")
                    duplicate_count += 1
                else:
                    seen_file_sizes[file_size] = filename
                    
        except Exception as e:
            print(f"❌ [CORRUPT FILE] Cannot decode bitstream for '{filename}'. Error: {e}")
            corrupt_files_found += 1

# --- SUMMARY PRODUCTION METRICS DASHBOARD ---
print("\n" + "="*50)
print("📊 FINAL DATA VALIDATION SUMMARY REPORT")
print("="*50)
print(f"✅ Total Asset Footprint Scanned : {total_images_scanned}")
print(f"❌ Completely Corrupt Files Found: {corrupt_files_found}")
print(f"⚠️ Severe Dimension Anomalies    : {bad_dimensions_found}")
print(f"👥 Redundant Duplicates Detected : {duplicate_count}")
print("="*50)

if corrupt_files_found > 0:
    print("🛑 STATUS: REJECTED. Please delete or fix corrupt assets before training.")
else:
    print("🟢 STATUS: PASSED. Your datasets are physically healthy and safe for model consumption.")
print("="*50)