# msc-dissertation
This repository contains my Master of Science Dissertation, in the programme Urban Spatial Science at The Bartlett Centre of Spatial Analysis (CASA) in the University College London (UCL).

The title of the dissertation is "The X-Minute City Paradox in São Paulo: local accessibility from everyday travel origins"

Data Wrangling for the São Paulo 2023 Origin Destination Survey:

1. Municipality filter: NumeroMuni == 36 isolates São Paulo municipality.
2. Age filter: only users from 18 to 65 years old.
3. Anchor filter: define anchor as trip origins that are related to the following trip purposes: 1-Work in industry, 2-Work in commerce, 3-Work in services, 4-School/Education and 8-Residence.
4. Included chaining trips (trips done between two anchors). 
5. CRS transformation to achieve WGS84 coordinates, not projected ones: microdata contains both projected (co_o_x, co_o_y in EPSG:22523) and WGS84 lat/lon (coord_x_o, coord_y_o). OSMnx and R5 require lat/lon. The script uses the WGS84 columns directly.

Architecture for the Local Accessibility Index calculation: 

1. Loads cleaned São Paulo 2023 Origin Destination Survey data (already filtered to SP municipality, age, anchors + chained trips).
2. Fetches OSM amenities for São Paulo, classifies them into the 10 categories, following Zhang et al. (2025). 
3. Builds the walk/bike street network from OSM, adds SRTM elevation grades, and converts edges into travel-time weights (minutes).
4. Snaps every trip origin to the nearest network node — one snap per individual row, no deduplication.
5. Queries nearest amenities via pandana for walk and bike: for every network node, it finds the cumulative minutes to the closest amenity in each category. Then it maps origins back to these times via their snapped node ID.
6. Runs R5 three times (walk+metro, walk+train, walk+bus) to get door-to-door public-transport times from each origin to the nearest amenity in each category.
7. Stores 50 raw travel-time columns (tt_walk_culture, tt_metro_healthcare, etc.) — all capped at 45 minutes.
8. Derives scores for 15, 20, and 30 minutes by checking tt <= threshold, applying the carbon weight, and taking the best weight per category.
9. Produces 30 category-score columns and 3 final index columns (pmc_index_15, pmc_index_30, pmc_index_45), while keeping every individual row intact.

   
