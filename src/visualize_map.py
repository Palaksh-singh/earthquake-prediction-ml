import argparse
import os
import pandas as pd
import folium

def main(args):
    assert os.path.exists(args.input), f"Processed file not found: {args.input}"
    df = pd.read_csv(args.input)

    # Center map near global center
    m = folium.Map(location=[20,0], zoom_start=2, tiles="cartodbpositron")

    # Add points
    for _, row in df.iterrows():
        lat, lon, mag, dep = row["Latitude"], row["Longitude"], row["Magnitude"], row["Depth"]
        popup = folium.Popup(f"Mag: {mag}, Depth: {dep} km", max_width=200)
        folium.CircleMarker(
            location=(lat, lon),
            radius=max(2, min(10, mag)),  # size roughly by magnitude
            fill=True,
            weight=0,
            fill_opacity=0.7
        ).add_child(popup).add_to(m)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    m.save(args.output)
    print(f"[OK] Map saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed.csv")
    parser.add_argument("--output", default="assets/earthquakes_map.html")
    main(parser.parse_args())
