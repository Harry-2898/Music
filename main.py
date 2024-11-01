import findmusic
import pandas as pd

def find_data_to_file(genre,max_tracks):
    data = findmusic.get_tracks_by_genre(genre , max_tracks)
    df = pd.DataFrame(data)
    df.to_csv(f'{genre}_tracks.csv', index=False)
    print(f"Dataset created with {len(df)} tracks for genre {genre}.")
