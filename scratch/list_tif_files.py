from remotezip import RemoteZip

url = "https://www.it.ubi.pt/DetReIDX_dataset/lusitano/test.zip"
with RemoteZip(url) as rz:
    namelist = rz.namelist()
    tif_files = [n for n in namelist if n.endswith('.tif')]
    print("Total .tif files:", len(tif_files))
    print("First 20 .tif files:")
    for name in tif_files[:20]:
        print(" ", name)
        
    jpg_files = [n for n in namelist if n.endswith('.jpg')]
    print("Total .jpg files:", len(jpg_files))
    
    # Check if there are other directories
    dirs = [n for n in namelist if n.endswith('/')]
    print("Directories:")
    for d in dirs:
        print(" ", d)
