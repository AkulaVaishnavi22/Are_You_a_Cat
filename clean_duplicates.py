import os

DATA_DIR = "data"
CLASS_NAMES = ['cats', 'dogs', 'humans']

print("🧹 Initializing Automated Duplicate Purge Pipeline...")

seen_file_sizes = {}
deleted_count = 0
total_scanned = 0

for folder in CLASS_NAMES:
    folder_path = os.path.join(DATA_DIR, folder)
    
    if not os.path.exists(folder_path):
        continue
        
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip directories or hidden system files
        if os.path.isdir(file_path) or filename.startswith('.'):
            continue
            
        total_scanned += 1
        file_size = os.path.getsize(file_path)
        
        if file_size in seen_file_sizes:
            # 🔥 DUPLICATE FOUND: Delete it safely from your hard drive
            print(f"🗑️ Deleting duplicate: {filename} (Matches original: {seen_file_sizes[file_size]})")
            os.remove(file_path)
            deleted_count += 1
        else:
            # First time seeing this image size, keep it!
            seen_file_sizes[file_size] = filename

print("\n" + "="*50)
print("✨ PURGE CYCLE COMPLETE")
print("="*50)
print(f"📦 Total files analyzed : {total_scanned}")
print(f"🔥 Duplicate files removed: {deleted_count}")
print(f"🛡️ Clean files remaining   : {total_scanned - deleted_count}")
print("="*50)