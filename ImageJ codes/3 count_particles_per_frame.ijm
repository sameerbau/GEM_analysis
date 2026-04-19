// 3 count_particles_per_frame.ijm
// Estimates particle density per movie by sampling N_SAMPLE evenly-spaced
// frames (default 10) rather than every frame.
//
// Rationale: density comparison between embryos (e.g. flagging over-linking
// artifacts like Em002) requires only a stable mean estimate. Per-frame
// count is stable in preblastoderm GEM movies (no division, slow bleaching),
// so 10 frames gives <5% SEM on the mean — sufficient to catch 3-5x outliers.
//
// Detection uses Find Maxima noise = max(3*frame_std, 10), approximating
// Mosaic ParticleTracker2D per/abs=0.3 absolute threshold.
//
// Outputs: particle_counts.csv        — sampled frames only
//          particle_counts_summary.csv — mean/std/median per embryo

N_SAMPLE = 10;   // frames to sample per movie — increase if you want finer stats

dir = getDirectory("Choose Source Directory");
run("Bio-Formats Macro Extensions");

detailPath  = dir + "particle_counts.csv";
summaryPath = dir + "particle_counts_summary.csv";

File.saveString("FileName,Frame,ParticleCount\n", detailPath);
File.saveString("FileName,MeanParticlesPerFrame,StdParticlesPerFrame,FramesSampled,TotalFrames,MedianParticlesPerFrame\n", summaryPath);

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
    nF = maxOf(slices, frames);

    // Build array of N_SAMPLE evenly-spaced frame indices (1-based)
    nSample = minOf(N_SAMPLE, nF);   // can't sample more frames than exist
    sampleFrames = newArray(nSample);
    for (s = 0; s < nSample; s++) {
        // Centre of each equal interval across the movie
        sampleFrames[s] = round(s * nF / nSample + nF / (2 * nSample)) + 1;
        sampleFrames[s] = minOf(sampleFrames[s], nF);  // clamp to valid range
    }

    counts = newArray(nSample);

    for (s = 0; s < nSample; s++) {
        f = sampleFrames[s];

        if (frames >= slices) {
            Stack.setFrame(f);
        } else {
            Stack.setSlice(f);
        }

        getStatistics(area, mean, min, max, std);
        noiseTol = maxOf(std * 3, 10);

        run("Find Maxima...", "noise=" + noiseTol + " output=[Point Selection]");

        if (selectionType() != -1) {
            getSelectionCoordinates(xc, yc);
            counts[s] = xc.length;
        } else {
            counts[s] = 0;
        }
        run("Select None");

        File.append(fname + "," + f + "," + counts[s] + "\n", detailPath);
    }

    // Summary statistics over sampled frames
    Array.getStatistics(counts, countMin, countMax, meanC, stdC);

    sorted = Array.copy(counts);
    Array.sort(sorted);
    mid = floor(nSample / 2);
    if (nSample % 2 == 1) {
        medC = sorted[mid];
    } else {
        medC = (sorted[mid - 1] + sorted[mid]) / 2.0;
    }

    File.append(fname + "," + d2s(meanC, 2) + "," + d2s(stdC, 2) + "," + nSample + "," + nF + "," + d2s(medC, 2) + "\n", summaryPath);
    print("  Total frames=" + nF + "  Sampled=" + nSample +
          "  Mean=" + d2s(meanC, 1) + "  Median=" + d2s(medC, 1) + " particles/frame");

    run("Close All");
}
