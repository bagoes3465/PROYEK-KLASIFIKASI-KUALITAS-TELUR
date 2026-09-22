import shutil
from pathlib import Path
import random
from tqdm import tqdm
import os

def split_dataset_three_way_robust(
    dataset_path, 
    train_ratio=0.7, 
    valid_ratio=0.15, 
    test_ratio=0.15, 
    seed=42
):
    """
    Split dataset dengan error handling yang robust
    """
    
    # Validasi rasio
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Total rasio harus = 1.0, sekarang = {total}")
    
    random.seed(seed)
    dataset_path = Path(dataset_path)
    
    # Path source
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    
    # Path destination
    valid_images = dataset_path / "valid" / "images"
    valid_labels = dataset_path / "valid" / "labels"
    test_images = dataset_path / "test" / "images"
    test_labels = dataset_path / "test" / "labels"
    
    # Buat folder destination
    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)
    test_images.mkdir(parents=True, exist_ok=True)
    test_labels.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SPLIT DATASET TRAIN → TRAIN + VALID + TEST")
    print("="*70)
    
    # Get semua image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG', '.JPEG']
    all_images = []
    
    print("\n🔍 Scanning images...")
    for ext in image_extensions:
        found = list(train_images.glob(f"*{ext}"))
        all_images.extend(found)
        if found:
            print(f"   Found {len(found)} files with extension {ext}")
    
    total_images = len(all_images)
    
    if total_images == 0:
        print(f"❌ Tidak ada gambar ditemukan di {train_images}")
        return
    
    print(f"\n📊 Total images: {total_images}")
    
    # Shuffle dan split
    random.shuffle(all_images)
    
    train_end = int(total_images * train_ratio)
    valid_end = train_end + int(total_images * valid_ratio)
    
    train_images_list = all_images[:train_end]
    valid_images_list = all_images[train_end:valid_end]
    test_images_list = all_images[valid_end:]
    
    print(f"\n📂 Train images: {len(train_images_list)} ({train_ratio*100:.0f}%)")
    print(f"📂 Valid images: {len(valid_images_list)} ({valid_ratio*100:.0f}%)")
    print(f"📂 Test images:  {len(test_images_list)} ({test_ratio*100:.0f}%)")
    
    # Function untuk move files dengan error handling
    def move_files_safe(file_list, dest_img_dir, dest_lbl_dir, split_name):
        print(f"\n🔄 Memindahkan {len(file_list)} files ke {split_name}...")
        
        moved_images = 0
        moved_labels = 0
        failed_moves = []
        missing_labels = []
        
        for img_path in tqdm(file_list, desc=f"Moving {split_name}"):
            try:
                # Cek apakah file masih ada
                if not img_path.exists():
                    failed_moves.append((str(img_path), "File tidak ditemukan"))
                    continue
                
                # Destination path
                dest_img = dest_img_dir / img_path.name
                
                # Cek apakah destination sudah ada
                if dest_img.exists():
                    print(f"\n⚠️  File sudah ada di destination: {img_path.name}")
                    continue
                
                # Move image dengan try-except
                try:
                    shutil.move(str(img_path), str(dest_img))
                    moved_images += 1
                except Exception as e:
                    # Jika move gagal, coba copy lalu delete
                    try:
                        shutil.copy2(str(img_path), str(dest_img))
                        os.remove(str(img_path))
                        moved_images += 1
                    except Exception as e2:
                        failed_moves.append((str(img_path), str(e2)))
                        continue
                
                # Move label
                label_name = img_path.stem + ".txt"
                label_path = train_labels / label_name
                
                if label_path.exists():
                    dest_label = dest_lbl_dir / label_name
                    try:
                        shutil.move(str(label_path), str(dest_label))
                        moved_labels += 1
                    except Exception as e:
                        try:
                            shutil.copy2(str(label_path), str(dest_label))
                            os.remove(str(label_path))
                            moved_labels += 1
                        except:
                            pass  # Label gagal move tidak fatal
                else:
                    missing_labels.append(img_path.name)
            
            except Exception as e:
                failed_moves.append((str(img_path), str(e)))
                continue
        
        print(f"   ✅ Images moved: {moved_images}/{len(file_list)}")
        print(f"   ✅ Labels moved: {moved_labels}/{len(file_list)}")
        
        if missing_labels:
            print(f"   ⚠️  Missing labels: {len(missing_labels)}")
        
        if failed_moves:
            print(f"   ❌ Failed moves: {len(failed_moves)}")
            print("\n   Failed files:")
            for path, error in failed_moves[:5]:  # Show first 5
                print(f"      - {Path(path).name}: {error}")
            if len(failed_moves) > 5:
                print(f"      ... dan {len(failed_moves)-5} lainnya")
        
        return moved_images, moved_labels, missing_labels, failed_moves
    
    # Move files ke valid dan test
    print("\n" + "="*70)
    print("MEMULAI PROSES SPLIT")
    print("="*70)
    
    all_failed = []
    
    _, _, missing_valid, failed_valid = move_files_safe(
        valid_images_list, valid_images, valid_labels, "VALID"
    )
    all_failed.extend(failed_valid)
    
    _, _, missing_test, failed_test = move_files_safe(
        test_images_list, test_images, test_labels, "TEST"
    )
    all_failed.extend(failed_test)
    
    # Verify hasil split
    print("\n" + "="*70)
    print("VERIFIKASI HASIL SPLIT")
    print("="*70)
    
    splits = {
        'TRAIN': dataset_path / "train",
        'VALID': dataset_path / "valid",
        'TEST': dataset_path / "test"
    }
    
    for split_name, split_path in splits.items():
        img_count = len(list((split_path / "images").glob("*.*")))
        lbl_count = len(list((split_path / "labels").glob("*.txt")))
        
        print(f"\n📁 {split_name}:")
        print(f"   Location: {split_path}")
        print(f"   Images: {img_count}")
        print(f"   Labels: {lbl_count}")
        
        if lbl_count < img_count:
            print(f"   ⚠️  Missing {img_count - lbl_count} labels")
    
    # Summary
    print("\n" + "="*70)
    if all_failed:
        print(f"⚠️  SPLIT SELESAI DENGAN {len(all_failed)} ERRORS")
        
        # Save error log
        error_log = dataset_path / "split_errors.log"
        with open(error_log, 'w', encoding='utf-8') as f:
            for path, error in all_failed:
                f.write(f"{path}\n{error}\n\n")
        print(f"📝 Error log saved to: {error_log}")
    else:
        print("✅ SPLIT SELESAI TANPA ERROR!")
    
    print("="*70)

if __name__ == "__main__":
    dataset_path = r"E:\PROYEK_KLASIFIKASI_TELUR\dataset\datav11"
    
    # Cek apakah folder valid/test sudah ada
    valid_path = Path(dataset_path) / "valid" / "images"
    test_path = Path(dataset_path) / "test" / "images"
    
    if valid_path.exists() or test_path.exists():
        print("⚠️  WARNING: Folder valid/test sudah ada!")
        print(f"   Valid: {valid_path.exists()}")
        print(f"   Test: {test_path.exists()}")
        
        response = input("\nApakah Anda ingin melanjutkan? Ini akan menambah file ke folder existing. (y/n): ")
        if response.lower() != 'y':
            print("❌ Split dibatalkan")
            exit()
    
    # Run split
    split_dataset_three_way_robust(
        dataset_path,
        train_ratio=0.70,
        valid_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )