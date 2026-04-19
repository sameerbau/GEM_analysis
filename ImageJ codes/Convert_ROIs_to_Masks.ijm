// This macro is to convert an ROI zip files with multiple ROIs to individual mask image files and 
// save them in an output folder

//Select Directories
input = getDirectory("Choose Source Directory");
output = getDirectory("Choose Destination Directory");

//===============================

run("Bio-Formats Macro Extensions"); // need this to open images with the special importer that does not have the popup window
close("*")

//Looping function through the files in a directory

list = getFileList(input);
for (i=0; i<list.length; i++) 
{
    if (endsWith(list[i], ".nd2")) 
    	{Ext.openImagePlus(input+list[i]);
    	action(input, output, list[i]);}
}
        
function action(input, output, filename) {
	roiManager("reset")
	roiManager("Open",input + list[i] + "_ROI.zip");
	N = roiManager("count");
	for (j=0; j<N; j++)
	{
		roiManager("Select",j);
		run("Create Mask");
		saveAs("Tiff",output + list[i] + "_" + j + "_mask.tif");
	}
	close("*");
}

