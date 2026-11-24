# README-ORIGINAL.md is here for legacy purposes. Its information is quite outdated. See below for up-to-date help.

If the root folder (the folder this .md file is in) has moved or has been renamed, delete the venv folder if it exists and follow the first time installation instructions.

## REQUIREMENTS:
* Windows 10 or later (other systems may work but aren't tested)
* Python 3.11.x or 3.12.x
* Git 2.45.2 or later
* Hydrus Network with API key
* A willingness to google stuff when things go wrong


## FOR FIRST TIME INSTALLATION:
1. Open the root folder (folder with this .md file) in cmd and run `python -m venv venv`
2. Run `venv\Scripts\pip install -r requirements.txt`

`run.bat` and `dry-run.bat` use the `venv` folder automatically, so there's no need to activate it yourself.

## DOWNLOADING THE MODELS:
This edit of the tool uses local copies of the models instead of downloading them from the internet when it runs. This means you will need to download the models yourself. Each folder in the `model` folder has a .txt file that says where to get that model and which files you need (usually `model.onnx` and `selected_tags.csv`).

## INFO.JSON:
In the pre-made model folders are info.json files. These tell the tagger which tagging model is being loaded and what it can do. This is important for certain functions such as tagging only the content rating (safe, explicit, etc.). If you want to use a new model that doesn't have a folder by default, make a new folder with the model's .onnx and .csv file in it, then create an info.json with the correct info for it.

## HYDRUS PREPARATION:
1. In Hydrus, navigate to services > manage services
2. Click "add" and select "local tag service"
3. Name your service "A.I. Tags" (or whatever you want)
4. Click apply, then click apply in manage services
5. Now navigate to services > review services
6. Navigate to local > client api
7. Click "add", select "manually"
8. Name the new api something identifiable like "AI tagging API"
9. Click the check box next to the "edit file tags" and "search and fetch files" permissions
10. Click apply, you can close review services now as well

While optional, it's recommended to migrate tag parents and tag siblings from the PTR to the new tag service to ensure consistent tag naming, as well as more detailed tagging.
To do so:

1. Navigate to tags > migrate tags
2. Under content, select "tag parents"
3. Under source, select "public tag repository"
4. Under filter, ensure it says "current" and under action ensure it says "add"
5. Under destination, select "A.I. Tags" (or whatever you named the new service)
6. Click "Go!" and wait for it to finish
7. Repeat steps 2 through 6, but under content select "tag siblings"

## CONFIGURATION:
1. Copy the example config: in the root folder, run `copy config.yaml.example config.yaml` in cmd
2. Edit config.yaml. At minimum set your Hydrus API `token`.
   - `tag_service` defaults to "A.I. Tags". If you named your tag service something else, change it here.
   - config.yaml is git-ignored, so your token stays on your PC and won't be committed.

Every setting is explained in `config.yaml.example`, and the values in it are the defaults. The ones you're most likely to change:

| Setting | What it does | Default |
|---|---|---|
| `input` | The text file that lists what to tag | `input.txt` |
| `mode` | `auto`, `tags` or `hashes` (how to read the input file) | `auto` |
| `model` | Model folder inside `model` | `wd-v1-4-vit-tagger-v2` |
| `threshold` | Lowest score (0 to 1) a tag needs to be kept | `0.35` |
| `limit` | Stop after this many files are tagged. 0 = no limit | `0` |
| `skip_tagged` | Skip files this model has already tagged | `false` |
| `dry_run` | Do everything except write to Hydrus | `false` |
| `retry` | Carry on by itself after a crash (see below) | `false` |
| `verbose` | Show every detail in the console | `false` |
| `tag_service` | Name of the local tag service to write to | `A.I. Tags` |
| `tag_service_key` | Pick the tag service by key instead (if two have the same name) | not set |
| `max_tags` | Most content tags added to one file | `150` |
| `request_timeout` | Seconds to wait for Hydrus to answer | `600` |
| `cpu` | `true` = run the model on the processor, `false` = graphics card | `false` |

Flags typed after the command (or after `run.bat`) override config.yaml for that one run, for example `run.bat --limit 10`.

## .BAT FILES:
There are two .bat files, and they are identical except for one flag:
- `run.bat` tags the files listed in your input file.
- `dry-run.bat` does everything except write tags to Hydrus. What would have been sent goes to `dryrun.log`.

Both read config.yaml, so you shouldn't need to edit them. You can also drag a text file onto either one to use that file instead of the `input` setting.

# Now you should be prepared and ready to start automatically tagging files.
I recommend starting with `dry-run.bat` and a small batch (for example `limit: 10`, or a search with `system:limit is 10`), then a real run on that same small batch. Make sure the tags are being put into your AI tagging service, that tag siblings/parents are displaying correctly, and that the file is properly tagged as being AI tagged. If all is well, you can proceed to mass tagging.

## (OPTIONAL) ENABLING GPU USE:
`requirements.txt` installs onnxruntime-gpu, but it needs NVIDIA's CUDA and cuDNN libraries to actually use the graphics card. Without them the tool still works on the processor, just slower. When a batch starts, the console shows which one is in use: `GPU (CUDA)` or `CPU (GPU not available)`.

The current models, such as [wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) or [wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3), need CUDA 12.x and cuDNN 9.x.

> [!WARNING]
> This was tested with CUDA 12.6.3 and cuDNN 9.5.1. For other versions, check the [onnxruntime requirements table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) to find a combination that matches your onnxruntime-gpu version.

- Install [CUDA Toolkit 12.6.3](https://developer.nvidia.com/cuda-toolkit-archive). You only need the runtime libraries, so you can uncheck the other install options to keep the installation small.
- Check that the CUDA installer added these folders to your Windows PATH:
	- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
	- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp`
- Install [cuDNN 9.5.1](https://developer.nvidia.com/cudnn-archive). By default it installs libraries for both CUDA 11.8 and CUDA 12.6; you can uncheck 11.8 if you don't need it.
	- **VERY IMPORTANT!** The installer adds only the `\bin\` folder to your Windows PATH, which is incorrect!
	- Edit the Windows PATH variable to match your CUDA version, so it looks like `C:\Program Files\NVIDIA\CUDNN\v9.5\bin\12.6`. Please **double check** this to avoid issues.
- Make sure you have the **Visual C++ Redistributables** installed. A decent all-in-one package can be found [here](https://github.com/abbodi1406/vcredist) if you don't want to install each one manually.
- `cpu: false` (the default) uses the graphics card when it can.

## FOR NORMAL USE:
1. Keep Hydrus OPEN.
2. Put what you want tagged in `input.txt` (copy `input.txt.example` to start). Use one of these, one per line:
	- **Hydrus search tags**, like a search page in Hydrus. A file must match all of them. System predicates work the same way they do in the Hydrus client. For example:
		`creator:some_artist`
		`system:filetype is image`
	- **File hashes**. In Hydrus, right click on files and navigate to share > copy > hash/hashes > sha256, then paste. The `sha256:` prefix is ignored automatically.
3. Run `dry-run.bat` to check, then `run.bat`.

The tool works out by itself whether the file holds tags or hashes. Blank lines and lines starting with `#` are ignored. If a file mixes both kinds, it stops and asks you to set `mode: tags` or `mode: hashes`.

Every tagged file also gets a `rating:...` tag (if the model has ratings) and a marker tag naming the model, like `wd v1.4 vit v2 tagger ai generated tags`, so you can find or skip AI-tagged files later.

### Skipping already-tagged files (`skip_tagged`)
With search tags, the tool adds a negative search term for the current model's marker (for example `-wd eva02 large tagger v3 ai generated tags`), so Hydrus never returns those files. With hashes, each file is checked before it's tagged. The marker comes from the selected model's `info.json`, so it works for any model you have installed.

- This only skips files tagged by the model you are currently running. To also skip files tagged by an older model, add the inverse of its marker to your input file, for example `-wd v1.4 vit v2 tagger ai generated tags`. You can find the marker in that model's info.json (the modelname field).

### Stopping after N tagged files (`limit`)
`limit` stops the batch after that many files have been tagged. Skipped files don't count. If your search has no `system:limit`, the tool adds one (`system:limit is N*20`) so Hydrus doesn't return the entire database when you only want a small batch. If you already put a `system:limit` in your search, that one is kept.

### Session log and resume
- `session.log` in the root folder lists every file the batch has handled, each with a status (OK, or the reason it was skipped or failed). When the batch finishes cleanly, the OK statuses are removed, leaving bare hashes, which is handy for copy-and-pasting. Files that were skipped or failed keep their reason so you can look into them. These are typically corrupted files or files with an incorrect filetype.
- If a batch stops part way (closed window, Ctrl+C, crash, power cut), the next run carries on where it left off. A `session.complete` file marks a batch that finished, so the next run starts fresh.
- Before carrying on, the tool checks that the model, mode (tags or hashes), `ratings_only` and tag service are the same as when the batch started. If they're not, it stops without contacting Hydrus and tells you what changed, so one batch never mixes two kinds of tagging. Changing the input file, threshold or limit is fine; it just prints a note.
- `--skip-resume` ignores an unfinished batch and starts a new one.
- Dry runs use `dryrun.log` and never touch `session.log`.

### Retry (`retry`)
With `retry: true`, if the batch crashes (for example Hydrus stops answering or is restarted), the tool waits and carries on from `session.log`. It waits 3 seconds, then 6, 12 and so on, up to 60 seconds between tries. It gives up after 8 failures in a row with no file finished in between, so an error that happens every time can't loop forever. It never retries after Ctrl+C, a setup mistake, a safety check or during a dry run.

### Stopping with Ctrl+C
Ctrl+C stops the batch. Everything finished so far is saved in `session.log`, and the next run carries on from there.

### Video and animated files
Multi-frame files (video, GIF, etc.) are sampled rather than tagged as a single image. Up to five frames are taken from across the file and tagged, and the scores are added together. A tag has to score more than twice your threshold across those frames to be kept, so videos usually get fewer tags. If you want more (but possibly less accurate) tags for videos, lower your threshold. Frames are decoded on the processor with `av` (listed in `requirements.txt`). Flash files (`.swf`, `.fla`, `.flash`) are excluded automatically since they can't be decoded. Thanks to Assamber (PR #10) for this feature!

### Safety checks
- Tags only go to a **local** tag service. The built-in "my tags" service is refused unless you add `--allow-my-tags`, and tag repositories such as the PTR are never written to.
- Tags you deleted from a file in Hydrus are not added back.
- A file is only tagged if Hydrus really has it.
- Model tags that contain a colon, like `re:zero`, are sent so Hydrus keeps them whole instead of reading the first part as a namespace.
- At most `max_tags` content tags (150 by default) are added to one file, best scores first.
- Your API token is never printed or written to the logs.

### The console
By default the console shows a few lines about the run, one progress bar with counts and time left, a short line for each file that had a problem, and a summary at the end. Colours: blue for the startup lines, yellow for problems and notices, red for errors, green for the summary. There are no colours when the output goes to a file or `NO_COLOR` is set. For every detail (debug lines, one line per file), use `--verbose` or `verbose: true`. `privacy: false` also prints each file's tags.

## Running in CLI
`run.bat` runs `python -m wd-hydrus-tagger`. With no command, that runs a batch using config.yaml. Every setting can also be given as a flag. To see all of them:
```bash
python -m wd-hydrus-tagger --help
```

### Evaluate a file
Tags one image on disk and prints the tags. Nothing is sent to Hydrus.
```bash
python -m wd-hydrus-tagger evaluate /path/to/file
```

### Evaluate a file in Hydrus
```bash
python -m wd-hydrus-tagger evaluate-api 123_your_files_hash_456
```

### Evaluate a number of files in Hydrus
```bash
python -m wd-hydrus-tagger my-list.txt --limit 10
```
`evaluate-api-batch` still works as another name for this, so older scripts keep working.

Add `--help` after any command to see its options.

## RANDOM INFO:
Everything has been confirmed to work on Windows 10. I've also read about other people getting everything (except GPU utilization) working on various linux distributions by changing the .bat files.
If you do run into problems on other operating systems, the culprit is likely in `interrogate.py`, specifically where path() is invoked, or something to do with the .bat files and how they're written.

The Z3D-E621-Convnext model doesn't tag content ratings but should provide much more accurate tags for furry artwork. `ratings_only: true` (or `--ratings-only`) with a WD model adds only the rating tag, no content tags. It never removes tags already on the file. So for furry art you can run the e621 model first, then a WD model with `--ratings-only` to add the rating.

My personal workflow is to have a hydrus search page that excludes all AI tags, filters to only PNG/JPG/WEBP images, and only images with less than 10 tags already (to avoid tagging already well-tagged files.) I also limit the results to around 2048 images, since that takes roughly an hour to process on my system.
After tagging EVERYTHING with a WD model, there will be some files that have been tagged with "furry" or "female furry".
I then have another search page that collects all these newly identified furry files, and I run the e621 model on them after a sufficient amount is identified.

If you prefer a different model, create a new folder in the /model/ folder, put your .onnx and .csv file in there, then copy and paste an info.json from another folder and edit it appropriately. Then set `model:` in config.yaml (or use `--model`) to the new folder's name.

You can send each model's tags to its own tag service if you'd like: create a new tag service and set `tag_service` (or use `--tag-service`). This might be useful if you want to delete wd-hydrus-tagger tags from furry files, or delete a tag service if/when a new, better tagging model comes out so you can re-tag everything.

As of the latest version of this tagging tool, the way that the "ai generated tags" tags are made is different, meaning they very likely won't match the tags given to files processed by older versions. I recommend sibling-ing the previous tag to the newly created one, or vice versa.

YOU MUST REBUILD THE VENV IF YOU MOVE OR RENAME THIS FOLDER

# CREDITS:
- SmilingWolf for the WD tagging models as well as some simple code to detect Kaomojis. (https://huggingface.co/SmilingWolf)
- Abtalerico for the well made original tool that I poorly edited to make this. (https://github.com/abtalerico/wd-hydrus-tagger) (deleted, copy can be found at  https://github.com/Garbevoir/wd-e621-hydrus-tagger/tree/main)
- Garbevoir for the wd-e621-hydrus-tagger edit this fork builds on, including the local model folders with info.json, E621 model support, ratings-only tagging, tag service selection and the setup guide this README grew from. (https://github.com/Garbevoir/wd-e621-hydrus-tagger)
- Zack3d (furzacky) for the E621 tagging model. (https://discord.com/channels/754509198674362388/1065785788698218526) (https://discord.gg/BDFpq9Yb7K)
- Assamber for the video and animated file sampling support! (https://github.com/Assamber/wd-e621-hydrus-tagger)
- This moderator on the Nvidia developer forums for providing safe downloads for zlibwapi.dll. (https://forums.developer.nvidia.com/t/could-not-load-library-cudnn-cnn-infer64-8-dll-error-code-193/218437/16)
- Hydrus Dev for developing and improving hydrus nonstop for several years, the real G.O.A.T. (https://hydrusnetwork.github.io/hydrus/index.html)
