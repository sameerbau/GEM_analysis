//Select Directories
dir1 = getDirectory("Choose Source Directory ");
dir2 = dir1;
//===============================
setBatchMode(true);
run("Bio-Formats Macro Extensions");

//Looping function through the files in a directory
list = getFileList(dir1);
for (i=0; i<list.length; i++) 
{
    if (endsWith(list[i], ".nd2")) 
    {
        // Open image only once
        Ext.openImagePlus(dir1+list[i]);
        
        // Crop operation
        makeRectangle(10, 10, 998, 998);
        run("Crop");
        run("Grays");
        cropFileName = list[i] + "_crop";
        saveAs("Tiff", dir2 + cropFileName);
        croppedImageID = getImageID();
        
        // Track particles on the cropped image
        selectImage(croppedImageID);
        run("Duplicate...", "duplicate channels=2");
        GEM=getImageID();
        selectImage(GEM);
        SingleParticleTracking();
        
        // Rename trajectory files
        origFile_csv = dir1 + "Traj_" + cropFileName + ".csv";
        newFile_csv = dir2 + "Traj_" + replace(list[i], ".tif", ".csv");
        
        // Wait for file to be created
        while(!File.exists(origFile_csv))
        {
            wait(1000);  // Delays (sleeps) for n milliseconds
        }
        wait(1000);
        
        s = File.rename(origFile_csv, newFile_csv);
    }
}
setBatchMode(false);

function SingleParticleTracking() 
{
    // GEM
    run("Particle Tracker 2D/3D", "radius=3 cutoff=0 per/abs=0.3 link=1 displacement=5 dynamics=Brownian");
    run("Close All");
}