from py_compile import main

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score



# DBSCAN Clustering and Visualization Function
# This function takes in the parameters for DBSCAN and the GeoDataFrame of points,
def dbscan_plot(eps_num, samples, gdf_points):
    

    # Fitting the DBSCAN Model
    # Assigning the parameters to the ones set earlier
    clustering = DBSCAN(eps=eps_num, min_samples=samples).fit(gdf_points[['Lat', 'Lon']])

    # Retrieving the labels
    dbscan_labels = clustering.labels_


    plt.figure(figsize=(8, 8))

    # Palette that distinguishes the clusters based on their labels
    unique_labels = set(dbscan_labels)
    colors = sns.color_palette("husl", len(unique_labels))

    # Plotting the results
    sns.scatterplot(
        x=gdf_points['Lon'],
        y=gdf_points['Lat'],
        hue=dbscan_labels,
        palette=colors,
        legend="full",
        s=50,
        alpha=0.7
    )

    plt.title(f"DBSCAN Clustering (eps={eps_num}, min_samples={samples})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Cluster ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


    # Summary Statistics
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points (outliers): {n_noise}")


# Function to find the best parameters for DBSCAN using silhouette score
# Ensure Crs is in EPSG:3400 before using this function
def dbscan_best_parameters(gdf_points, max_eps, min_eps, step_eps, min_samples_range):


    best_eps = None
    best_min_samples = None



    # YOUR CODE HERE

    eps_range = np.linspace(min_eps, max_eps, int((max_eps - min_eps) / step_eps))

    results = pd.DataFrame(columns=["silhouette_score", "eps", "min_samples"])

    for ep in eps_range:
        for sample in min_samples_range:

            # Creating a DBSCAN with the dataset
            db_scan = DBSCAN(eps=ep, min_samples=sample)

            labels = db_scan.fit_predict(gdf_points[['Lat', 'Lon']])

            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 1:

                # Computing the silhouette scores
                score = silhouette_score(gdf_points[['Lat', 'Lon']], labels)
                
                # Storing the results
                results.loc[len(results)] = [score, round(ep,2), sample]


    # Getting the best scores
    best_index = results.loc[results["silhouette_score"].idxmax()]
    best_eps = best_index["eps"]
    best_min_samples = int(best_index["min_samples"])

    # retrain with the best parameters
    db_scan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    labels = db_scan.fit_predict(gdf_points[['Lat', 'Lon']])

    # print the best eps
    print("Best eps: ", best_eps)
    # print the best min_samples
    print("Best min_samples: ", best_min_samples)

    return best_eps, best_min_samples, labels


def dbscan_folium(n_clusters, dbscan_labels, gdf_points):
    # 1. Create a color palette for clusters (excluding noise)
    color_list = sns.color_palette("husl", n_clusters).as_hex()
    
    # 2. Add 'Grey' at the end specifically for noise (label -1)
    # This prevents an IndexError when row['cluster_label_dbscan'] is -1
    color_list.append("#808080") 

    map_center = [gdf_points['Latitude'].mean(), gdf_points['Longitude'].mean()]
    map_ = folium.Map(location=map_center, zoom_start=8, tiles="OpenStreetMap")

    gdf_points["cluster_label_dbscan"] = dbscan_labels

    for i in range(len(gdf_points)):
        row = gdf_points.iloc[i]
        cluster_idx = int(row['cluster_label_dbscan'])
        
        # Logic to pick color: if cluster_idx is -1, use the last color (Grey)
        marker_color = color_list[cluster_idx] if cluster_idx != -1 else color_list[-1]
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7,
            popup=f"Cluster: {cluster_idx}<br>Encounter: {row.get('Encounter Type', 'N/A')}"
        ).add_to(map_)

    return map_ # MUST return the map object


# Function to compute statistics of encounters in each cluster
def statistics(labels, cluster_label, gdf_points):
    

    #Place holder dictionary
    count_dict = {}

    for en in labels:
            for cl in cluster_label:
                count = len(gdf_points[(gdf_points['cluster'] == cl) & 
                                        (gdf_points['Encounter Type'] == en)])
                count_dict[(en, cl)] = count
                print(f"Number of {en} encounters in cluster {cl}: {count}")

    print("\nSummary of counts:")
    for key, value in count_dict.items():
        print(f"{key}: {value}")


    for key, value in count_dict.items():
        percentage = (count_dict[key] / len(gdf_points[gdf_points["cluster"] == key[1]])) * 100 if len(gdf_points[gdf_points["cluster"] == key[1]]) > 0 else 0
        print(f"Percentage {key[0]} encounters in cluster {key[1]}: {percentage:.2f}%")


if __name__ == "main":
    main()

