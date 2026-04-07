import os
import wave

def getSortedFilesByDate(folder, suffix):
    files = [f for f in os.listdir(folder) if f.endswith(suffix)]
    
    # Sort by modification time (oldest newest)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    
    return files


def getTotalFrames(fileList, folder):
    total = 0
    for file in fileList:
        with wave.open(os.path.join(folder, file), 'rb') as wf:
            total += wf.getnframes()
    return total


def joinWavFilesStreaming(inputFolder, outputFile, suffix="1008.wav", chunkSize=1024 * 1024):

    wavFiles = getSortedFilesByDate(inputFolder, suffix)

    if not wavFiles:
        print("No matching WAV files found.")
        return

    print("Files (oldest -> newest by save date):")
    for f in wavFiles:
        fullPath = os.path.join(inputFolder, f)
        print(f"{f}  |  {os.path.getmtime(fullPath)}")

    print("\nCalculating total duration...")
    totalFrames = getTotalFrames(wavFiles, inputFolder)

    firstFilePath = os.path.join(inputFolder, wavFiles[0])

    with wave.open(firstFilePath, 'rb') as wf:
        params = wf.getparams()
        sampleRate = wf.getframerate()

    totalSeconds = totalFrames / sampleRate
    processedFrames = 0

    with wave.open(outputFile, 'wb') as out:
        out.setparams(params)

        for file in wavFiles:
            filePath = os.path.join(inputFolder, file)
            print(f"\nProcessing: {file}")

            with wave.open(filePath, 'rb') as wf:

                if (
                    wf.getnchannels() != params.nchannels or
                    wf.getsampwidth() != params.sampwidth or
                    wf.getframerate() != params.framerate or
                    wf.getcomptype() != params.comptype
                ):
                    raise ValueError(f"Format mismatch in file: {file}")

                while True:
                    frames = wf.readframes(chunkSize)
                    if not frames:
                        break

                    out.writeframes(frames)

                    # Update progress
                    processedFrames += len(frames) // params.sampwidth
                    processedSeconds = processedFrames / sampleRate
                    progress = processedSeconds / totalSeconds * 100

                    print(
                        f"\rProgress: {progress:6.2f}% | "
                        f"{processedSeconds/3600:.2f}h / {totalSeconds/3600:.2f}h",
                        end=""
                    )

    print(f"\n\nDone. Output saved to: {outputFile}")



# ==== USAGE ====
inputFolder = r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\completeSound"
outputFile = r"C:\Users\Christian Lykke\Documents\Skole\Aalborg Universitet\CEAIVS8\Projekt\Lydfiler\completeSound\completeBackgroundNoise28032026__1008.wav"

joinWavFilesStreaming(inputFolder, outputFile)