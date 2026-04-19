// 3 count_particles_per_frame.ijm
// Count detected particles per frame for every *crop.tif in a folder.
// Uses Find Maxima with noise = max(3 * frame_std, 10), approximating
// Mosaic ParticleTracker2D detection (radius=3, per/abs=0.3 absolute).
// Outputs: particle_counts.csv and particle_counts_summary.csv

dir = getDirectory("Choose Source Directory");
run("Bio-Formats Macro Extensions");

detailPath  = dir + "particle_counts.csv";
summaryPath = dir + "particle_counts_summary.csv";

File.saveString("FileName,Frame,ParticleCount\n", detailPath);
File.saveString("FileName,MeanParticlesPerFrame,StdParticlesPerFrame,TotalFrames,MedianParticlesPerFrame\n", summaryPath);

list = getFileList(dir);
for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], "crop.tif")) {
        countParticlesInFile(list[i]);
    }
}

print("Done.");
print("Per-frame counts : " + detailPath);
print("Summary          : " + summaryPath);

function countParticlesInFile(fname) {
    print("Processing: " + fname);
    Ext.openImagePlus(dir + fname);

    getDimensions(width, height, channels, slices, frames);
    // Time-lapse may load as frames>1 or slices>1 depending on Bio-Formats metadata
    nF = maxOf(slices, frames);

    counts = newArray(nF);

    for (f = 1; f <= nF; f++) {
        if (frames >= slices) {
            Stack.setFrame(f);
        } else {
            Stack.setSlice(f);
        }

        // Noise tolerance approximating Mosaic per/abs=0.3 absolute threshold:
        // applied to each frame independently so it adapts to intensity drift
        getStatistics(area, mean, min, max, std);
        noiseTol = maxOf(std * 3, 10);

        run("Find Maxima...", "noise=" + noiseTol + " output=[Point Selection]");

        if (selectionType() != -1) {
            getSelectionCoordinates(xc, yc);
            counts[f - 1] = xc.length;
        } else {
            counts[f - 1] = 0;
        }
        run("Select None");

        File.append(fname + "," + f + "," + counts[f - 1] + "\n", detailPath);
    }

    // Summary statistics
    Array.getStatistics(counts, countMin, countMax, meanC, stdC);

    sorted = Array.copy(counts);
    Array.sort(sorted);
    mid = floor(nF / 2);
    if (nF % 2 == 1) {
        medC = sorted[mid];
    } else {
        medC = (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    File.append(fname + "," + d2s(meanC, 2) + "," + d2s(stdC, 2) + "," + nF + "," + d2s(medC, 2) + "\n", summaryPath);
    print("  Frames=" + nF + "  Mean=" + d2s(meanC, 1) + "  Median=" + d2s(medC, 1) + " particles/frame");

    run("Close All");
}
