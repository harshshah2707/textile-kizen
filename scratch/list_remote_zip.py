from remotezip import RemoteZip

url = "https://www.it.ubi.pt/DetReIDX_dataset/lusitano/test.zip"
print("Connecting to remote zip...")
with RemoteZip(url) as rz:
    print("Remote zip loaded successfully.")
    namelist = rz.namelist()
    print("Total files in zip:", len(namelist))
    
    # Print the first 50 files
    print("\nFirst 50 files:")
    for name in namelist[:50]:
        print(name)
        
    # Categorize and count file extensions
    exts = {}
    for name in namelist:
        ext = name.split('.')[-1].lower() if '.' in name else 'no_ext'
        exts[ext] = exts.get(ext, 0) + 1
    print("\nFile extension counts:")
    for ext, count in exts.items():
        print(f"  .{ext}: {count}")
