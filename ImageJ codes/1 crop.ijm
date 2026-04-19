//Select Directories
input = getDirectory("Choose Source Directory");
output = input;
//getDirectory("Choose Destination Directory");

//===============================
setBatchMode(true);
run("Bio-Formats Macro Extensions"); // need this to open images with the special importer that does not have the popup window

list = getFileList(input);
for (i = 0; i < list.length; i++)
        action(input, output, list[i]);

function action(input, output, filename) {
        Ext.openImagePlus(input+list[i]); //to open images without the bioformats importer
        run("Duplicate...", "duplicate channels=2");
		//makeRectangle(0, 6, 512, 499);       
       // makeRectangle(0, 20, 1024, 983); // Parameter based on the opened image pixel sizes 
        dup=getImageID();
        selectImage(dup);
		makeRectangle(10, 10, 998, 998);
		//makeRectangle(5, 5, 500, 500);

        run("Crop");
        run("Grays");
        saveAs("Tiff", output + filename + "_crop");
        close();
}
setBatchMode(false);