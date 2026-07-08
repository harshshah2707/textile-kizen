from remotezip import RemoteZip

url = "https://www.it.ubi.pt/DetReIDX_dataset/lusitano/test.zip"
with RemoteZip(url) as rz:
    namelist = rz.namelist()
    
    defects_jpg = [n for n in namelist if n.startswith('test/defects/') and n.endswith('.jpg')]
    defects_tif = [n for n in namelist if n.startswith('test/defects/') and n.endswith('.tif')]
    non_defects_jpg = [n for n in namelist if n.startswith('test/non-defects/') and n.endswith('.jpg')]
    non_defects_tif = [n for n in namelist if n.startswith('test/non-defects/') and n.endswith('.tif')]
    
    print("Files in test/defects/:")
    print(f"  JPG: {len(defects_jpg)}")
    print(f"  TIF: {len(defects_tif)}")
    print("Files in test/non-defects/:")
    print(f"  JPG: {len(non_defects_jpg)}")
    print(f"  TIF: {len(non_defects_tif)}")
    
    # Are there any other paths that are files?
    other = [n for n in namelist if not n.endswith('/') and not n.startswith('test/defects/') and not n.startswith('test/non-defects/')]
    print(f"Other files: {len(other)}")
    for o in other:
        print(" ", o)
