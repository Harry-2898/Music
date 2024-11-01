import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

# Set up authentication
client_id = '8dfd048c32d74c29b6a80dbe58e7e4f3'      
client_secret = '7472442fd5b0423bb68f4c95f44ef00d'  
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Function to get tracks for a specific genre
def get_tracks_by_genre(genre, limit=50, max_tracks=500):
    tracks = []
    offset = 0
    
    while len(tracks) < max_tracks:
        try:
            results = sp.search(q=f'genre:"{genre}"', type='track', limit=limit, offset=offset)
            items = results.get('tracks', {}).get('items', [])
            
            for item in items:
                track_info = {
                    'track_name': item['name'],
                    'artist_name': item['artists'][0]['name'],
                    'album_name': item['album']['name'],
                    'release_date': item['album']['release_date'],
                    'popularity': item['popularity'],
                    'track_id': item['id']
                }
                tracks.append(track_info)
                
                # Stop if we've reached the max tracks
                if len(tracks) >= max_tracks:
                    break
            
            offset += limit
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return tracks

if __name__ == '__main__':
    # Specify the genre and get the data
    genre = input("What Genre do you want? ")  
    data = get_tracks_by_genre(genre, max_tracks=100)
    df = pd.DataFrame(data)
    df.to_csv(f'{genre}_tracks_dataset.csv', index=False)
    print(f"Dataset created with {len(df)} tracks for genre '{genre}'.")
