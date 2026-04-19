//Select Directories
dir1 = getDirectory("Choose Source Directory ");
dir2 = dir1;
//getDirectory("Choose Destination Directory Results ");
//===============================


run("Bio-Formats Macro Extensions");

//Looping function through the files in a directory

list = getFileList(dir1);
for (i=0; i<list.length; i++) 
{
    if (endsWith(list[i], "crop.tif")) 
    	{Ext.openImagePlus(dir1+list[i]);
    	SingleParticleTracking();}
}

//all files set to run
//need to check for when files are finished - rename them as needed
for (i=0; i<list.length; i++) 
{
    if (endsWith(list[i], ".tif")) 
    {
    	
		origFile_csv = dir1 + "Traj_" + list[i] + ".csv";  // Only csv files of trajectory information (not txt files of particle detection in each frame) are used in the subsequent analysis
		newFile_csv = dir2 + "Traj_" + replace(list[i], ".tif", ".csv");
		while(!File.exists(origFile_csv))
		{
			wait(1000);  // Delays (sleeps) for n milliseconds
		}
		wait(1000);

		s=File.rename(origFile_csv, newFile_csv);    /// IF FILE ALREADY EXISTS IT will OVERWRITE! 

//		origFile_txt = dir1 + "Traj_" + list[i] + ".txt";
//		newFile_txt = dir2 + "Traj_" + replace(list[i], ".tif", ".txt");
//		s=File.rename(origFile_txt, newFile_txt);    /// IF FILE ALREADY EXISTS IT will OVERWRITE!
    	
    }
}

function SingleParticleTracking() 
{
	//waitForUser("W1");	//Halts the macro and displays "W1" in a dialog box. The macro proceeds when the user clicks "OK".
    // GEM
	run("Particle Tracker 2D/3D", "radius=3 cutoff=0 per/abs=0.3 link=1 displacement=5 dynamics=Brownian"); //(Confocal embryo GEM, pixel size=0.0943670um)
    run("Close All");
    
}
