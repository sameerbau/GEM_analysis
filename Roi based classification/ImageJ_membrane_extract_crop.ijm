  setBatchMode(true)
//Select Directories
input = getDirectory("Choose Source Directory");
output = input;

//===============================

run("Bio-Formats Macro Extensions"); // need this to run to open images with the special importer (see below)
									// that does not have the popup window

list = getFileList(input);			//a new variable containing the names of the files in the input variable
for (i = 0; i < list.length; i++)
        action(input, output, list[i]);

function action(input, output, filename) {
        Ext.openImagePlus(input+list[i]); //this is the line to open images without the bioformats importer
        a=getImageID();
		run("Duplicate...", "duplicate channels=1 frames=1");
		selectImage(a);
	
		close();
			//makeRectangle(10, 10, 490, 490);
			//makeRectangle(5, 5, 500, 500);
			makeRectangle(10, 10, 998, 998);
		run("Crop");
        saveAs("Tiff", output + filename + "_membrane_");
        close();
}