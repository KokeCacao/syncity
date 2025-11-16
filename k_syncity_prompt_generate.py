#!/usr/bin/env python3
# generate_files.py
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List

Grid = List[List[str]]  # 4 rows × 3 columns

def main(out: pathlib.Path, single: bool) -> None:
    if single:
        _prompts = {
            "city single": "a large city block tile",
            "medieval single": "a game tile, medieval style",
            "desert single": "a game tile, desert style",
            "cyberpunk single": "a game tile, cyberpunk style",
            "ancient rome single": "a game tile, ancient rome style",
            "minecraft single": "a game tile, minecraft style",
            "forest single": "a game tile, forest style",
            "ocean single": "a game tile, ocean style",
            "winter single": "a game tile, winter style",
            "lego single": "a game tile, lego style",
            "park single": "a game tile, park style",
            "amusement park single": "a game tile, amusement park style",
            "airport single": "a game tile, airport style",
            "college single": "a game tile, college campus style",
            "room single": "a game tile, modern living room",
        }
        prompts = {k: [[v]*3 for _ in range(4)] for k, v in _prompts.items()}
    else:
        prompts: Dict[str, Grid] = {
            "city": [
                ["a large city block with parks", "a large city block with skyscrapers", "a large city block with bridges and rivers"],
                ["a large city block with ponds and trees", "a large city block with parks", "a large city block near ocean"],
                ["a large city block with residential houses", "a large city block with residential houses", "a large city block coastal line"],
                ["a large city block with industry power plants", "a large city block, construction site", "a large city block, construction site near ocean"],
            ],
            "medieval": [
                ["a medieval tile with farmland fields and cottages", "a medieval tile with a hamlet of thatched houses", "a medieval tile with a village market square"],
                ["a medieval tile with a blacksmith and small workshops", "a medieval tile with an inn and stables", "a medieval tile with a guildhall and stalls"],
                ["a medieval tile with stone townhouses and narrow streets", "a medieval tile with a city wall segment and gate", "a medieval tile with a hilltop keep and bailey"],
                ["a medieval tile with a river, stone bridge and watermill", "a medieval tile with riverside docks and warehouses", "a medieval tile with a coastal harbor and lighthouse"],
            ],
            "desert": [
                ["a desert tile with dune fields and scattered cacti", "a desert tile with dunes and rocky outcrops", "a desert tile with an oasis and palm grove"],
                ["a desert tile with a caravanserai courtyard and stalls", "a desert tile with a mudbrick village and alleys", "a desert tile with a hilltop watchtower and walls"],
                ["a desert tile with a canyon and a small stone bridge", "a desert tile with a dry riverbed crossing (wadi)", "a desert tile with ancient ruins and broken columns"],
                ["a desert tile with a nomad tent camp and windbreaks", "a desert tile with a well and date grove", "a desert tile with a salt flat waystation and markers"],
            ],
            "cyberpunk": [
                ["a cyberpunk city tile with a street market and modular stalls", "a cyberpunk city tile with service alleys and exposed pipes", "a cyberpunk city tile with a plaza and neon billboard frames"],
                ["a cyberpunk city tile with a residential megablock entrance", "a cyberpunk city tile with a skybridge between towers", "a cyberpunk city tile with a monorail platform and tracks"],
                ["a cyberpunk city tile with a research lab facade and loading bay", "a cyberpunk city tile with a rooftop greenhouse garden", "a cyberpunk city tile with an energy substation and conduits"],
                ["a cyberpunk city tile with a nightclub frontage and signage", "a cyberpunk city tile with a tech bazaar of modular kiosks", "a cyberpunk city tile with a security gate and turnstiles"],
            ],
            "ancient rome": [
                ["an ancient rome tile with forums and columned porticoes", "an ancient rome tile with aqueduct arches and road", "an ancient rome tile with a marketplace and stalls"],
                ["an ancient rome tile with villas and peristyle gardens", "an ancient rome tile with statues and fountains", "an ancient rome tile with public baths (thermae)"],
                ["an ancient rome tile with narrow insulae-lined streets", "an ancient rome tile with a triumphal arch and plaza", "an ancient rome tile with a fortified gate and walls"],
                ["an ancient rome tile with a stone bridge over a river", "an ancient rome tile with riverside warehouses and docks", "an ancient rome tile with a harbor quay and lighthouse"],
            ],
            "minecraft": [
                ["a minecraft terrain tile with rolling hills and trees", "a minecraft terrain tile with a river and a small bridge", "a minecraft terrain tile with a dense forest"],
                ["a minecraft terrain tile with a village and paths", "a minecraft terrain tile with farms and animal pens", "a minecraft terrain tile with a mine entrance"],
                ["a minecraft terrain tile with mountains and caves", "a minecraft terrain tile with a stone castle on a hill", "a minecraft terrain tile with a fortified outpost"],
                ["a minecraft terrain tile with a waterfall and pool", "a minecraft terrain tile with a coastal beach and pier", "a minecraft terrain tile with a volcanic peak and lava flows"],
            ],
            "forest": [
                ["a dense forest tile with tall trees and a small clearing", "a dense forest tile with a river and a wooden footbridge", "a dense forest tile with a timber cabin and porch"],
                ["a dense forest tile with a small pond and boardwalk", "a dense forest tile with mossy boulders and logs", "a dense forest tile with a wayfinding signpost"],
                ["a dense forest tile with a hiking trail and steps", "a dense forest tile with autumn foliage on the ground", "a dense forest tile with a waterfall and rock ledges"],
                ["a dense forest tile with a campfire ring and benches", "a dense forest tile with wildflower meadows and path", "a dense forest tile with rocky escarpment and lookout"],
            ],
            "ocean": [
                ["an ocean tile with open water and a navigation buoy", "an ocean tile with a small rocky islet and beacon", "an ocean tile with a lighthouse on a rocky promontory"],
                ["an ocean tile with a sandy beach and tide pools", "an ocean tile with a wooden pier and a moored ship", "an ocean tile with a coastal breakwater and markers"],
                ["an ocean tile with a reef and a small island grove", "an ocean tile with a stone seawall and stairs to water", "an ocean tile with sea stacks and arches"],
                ["an ocean tile with drift ice and small icebergs", "an ocean tile with larger icebergs and snow on shore", "an ocean tile with a polar research station on ice"],
            ],
            "winter": [
                ["a snowy winter tile with pine trees and a frozen lake", "a snowy winter tile with a timber cabin and chimney", "a snowy winter tile with snow-covered mountains"],
                ["a snowy winter tile with a sled trail and fence posts", "a snowy winter tile with a ski slope and lift pylons", "a snowy winter tile with bare trees and drifts"],
                ["a snowy winter tile with a stone bridge over a frozen river", "a snowy winter tile with an ice cave entrance", "a snowy winter tile with a snow-covered village square"],
                ["a snowy winter tile with a watchtower on a ridge", "a snowy winter tile with a frozen pond and footbridge", "a snowy winter tile with storage sheds and woodpiles"],
            ],
            "lego": [
                ["a lego city tile with buildings and roads", "a lego city tile with a park and playground", "a lego city tile with a train station"],
                ["a lego city tile with skyscrapers", "a lego city tile with a harbor and boats", "a lego city tile with a construction site"],
                ["a lego city tile with houses and gardens", "a lego city tile with a shopping mall", "a lego city tile with a sports stadium"],
                ["a lego city tile with a fire station", "a lego city tile with a police station", "a lego city tile with an airport"],
            ],
            "park": [
                ["a park tile with a playground and picnic area", "a park tile with a pond and ducks statue and pier", "a park tile with walking trails and signposts"],
                ["a park tile with a garden and flowers", "a park tile with a fountain", "a park tile with a sports field and bleachers"],
                ["a park tile with tall trees and benches", "a park tile with a gazebo", "a park tile with a bike path and racks"],
                ["a park tile with a dog park enclosure", "a park tile with a skate park bowl", "a park tile with an amphitheater stage"],
            ],
            "amusement park": [
                ["an amusement park tile with a roller coaster and ferris wheel", "an amusement park tile with a carousel", "an amusement park tile with food stalls and seating"],
                ["an amusement park tile with a haunted house", "an amusement park tile with a water ride flume", "an amusement park tile with a circus tent and ring"],
                ["an amusement park tile with lighting rigs and archways", "an amusement park tile with a bumper car pavilion", "an amusement park tile with a mini golf course"],
                ["an amusement park tile with a parade route and grandstand", "an amusement park tile with a performance stage and truss", "an amusement park tile with souvenir shops and kiosks"],
            ],
            "airport": [
                ["an airport tile with a runway and control tower", "an airport tile with airplanes on stands", "an airport tile with a terminal building"],
                ["an airport tile with luggage carts and service lanes", "an airport tile with a parking lot", "an airport tile with a hangar"],
                ["an airport tile with jet bridges", "an airport tile with a fuel truck bay and pumps", "an airport tile with a security checkpoint hall"],
                ["an airport tile with a baggage claim hall", "an airport tile with a customs inspection area", "an airport tile with a duty-free shop concourse"],
            ],
            "college": [
                ["a college campus tile with academic buildings and green spaces", "a college campus tile with a library", "a college campus tile with a student center"],
                ["a college campus tile with dormitories", "a college campus tile with a sports field", "a college campus tile with a cafeteria"],
                ["a college campus tile with a science lab", "a college campus tile with an art building", "a college campus tile with a music hall"],
                ["a college campus tile with a quad", "a college campus tile with a fountain", "a college campus tile with a statue"],
            ],
            "room": [
                ["a modern living room tile with a sofa, coffee table, and TV", "a modern living room tile with large windows and indoor plants", "a modern living room tile with a fireplace and mantle"],
                ["a modern living room tile with a bookshelf wall", "a modern living room tile with framed abstract art", "a modern living room tile with a patterned area rug"],
                ["a modern living room tile with a dining nook and table", "a modern living room tile with a minimalist console and storage", "a modern living room tile with layered ceiling lights"],
                ["a modern living room tile with a gaming/media setup", "a modern living room tile with a reading chair and floor lamp", "a modern living room tile with a bay window seating bench"],
            ],
        }

    suffix_default: str = "isometric, voxel art style"
    suffix_by_cat: Dict[str, str] = {
        "city": "isometric, voxel art style"
    }

    out.mkdir(parents=True, exist_ok=True)

    for category, grid in prompts.items():
        tiles: List[Dict[str, Any]] = [
            {"prompt": prompt, "x": x, "y": y}
            for y, row in enumerate(grid)
            for x, prompt in enumerate(row)
        ]
        suffix: str = suffix_by_cat.get(category, suffix_default)
        payload: Dict[str, Any] = {
            "tiles": tiles,
            "prompt": "{tile_prompt}, " + suffix,
        }
        out_path: pathlib.Path = out / f"{category.replace(' ', '_')}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-category tile JSON files.")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("instructions/4x3"), help="Output directory.")
    parser.add_argument("--single", action="store_true", help="Generate a single prompt version.")
    args = parser.parse_args()
    main(out=args.out, single=args.single)
