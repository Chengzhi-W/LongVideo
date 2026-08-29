# UltraVideo-Long 5K-20s locked selection

`selection_ids.csv` deterministically selects 5,000 unique clips from the
official `APRIL-AIGC/UltraVideo-Long` `long.csv` metadata:

- minimum duration: 20.0 seconds
- sampling seed: 42
- split: 4,900 train / 100 validation
- resolution used by the downloader: 960p archive
- official `long.csv` SHA-256:
  `6a14434c3c74b79a5ef3c8bf5f60fbd1b5f71cb27c1faea9300d0e372e961191`
- sorted-ID selection SHA-256:
  `540a4bc3b7ebcae94d9afb896b9986457326c375beb81a68d99545d4e0682804`

The compact file contains IDs and split assignments only. Captions and full
metadata are regenerated from the official dataset and are not redistributed.
