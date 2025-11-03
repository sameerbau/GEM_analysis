a=getImageID();
selectImage(a);

XY_image_name=getTitle();
dir=getDirectory("image");

roipath=dir+"/"+XY_image_name+"_rois.zip";
roiManager("save", roipath);


roiManager("reset");

selectImage(a);
close();
