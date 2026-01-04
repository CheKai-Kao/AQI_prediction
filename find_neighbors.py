import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix


def get_station_neighbors(csv_path='空氣品質監測站基本資料.csv', k=3):
    df = pd.read_csv(csv_path)
    
    last_station_name = df.iloc[-1]['sitename']
    print(f"[Info] 已排除資料表中最後一個測站不參與計算: {last_station_name}")
    df = df.iloc[:-1].reset_index(drop=True)  # 排除員林

    site_ids = df['siteid'].astype(int).values
    site_names = df['sitename'].values
    name_to_id = dict(zip(site_names, site_ids))
    
    coords_deg = df[['twd97lon', 'twd97lat']].values
    id_to_coord = dict(zip(site_ids, coords_deg))
    
    LON_TO_M = 102000.0  # 經度每度約 102 公里
    LAT_TO_M = 111000.0  # 緯度每度約 111 公里
    
    coords_meter = coords_deg.copy()
    coords_meter[:, 0] *= LON_TO_M
    coords_meter[:, 1] *= LAT_TO_M
    
    dist_mtx = distance_matrix(coords_meter, coords_meter)
    
    neighbor_dict = {}
    relative_pos_dict = {} 
    
    area_support_map = {
        "北部空品區": ["金門", "馬祖"],
        "竹苗空品區": ["金門", "馬祖", "頭份"],
        "中部空品區": ["金門", "馬祖", "頭份", "線西"],
        "雲嘉南空品區": ["金門", "馬祖", "線西", "臺西", "麥寮"],
        "高屏空品區": ["金門", "馬祖", "臺西", "麥寮", "前鎮"],
        "其他": ["金門", "馬祖"],
        "花東空品區": ["金門", "馬祖"],
        "宜蘭空品區": ["金門", "馬祖"],
    }
    
    for i, sid in enumerate(site_ids):
        sid = int(sid)
        current_area = df.iloc[i]['areaname']
        
        target_lon, target_lat = coords_deg[i]
        
        nearest_idx = np.argsort(dist_mtx[i])[1:k+1]
        spatial_neighbor_ids = site_ids[nearest_idx].tolist()
        
        support_names = area_support_map.get(current_area, [])
        support_ids = [name_to_id[n] for n in support_names if n in name_to_id]
        
        combined_set = set(spatial_neighbor_ids + support_ids)
        if sid in combined_set: combined_set.remove(sid)
        final_neighbors = list(combined_set)
        neighbor_dict[sid] = final_neighbors
        
        rel_features = []
        
        for nb_id in final_neighbors:
            nb_lon, nb_lat = id_to_coord[nb_id]
            
            dx_deg = nb_lon - target_lon
            dy_deg = nb_lat - target_lat
            
            dx_m = dx_deg * LON_TO_M
            dy_m = dy_deg * LAT_TO_M
            
            real_dist_m = np.sqrt(dx_m**2 + dy_m**2)
            
            dist_log = np.log10(real_dist_m + 1.0)
            
            angle = np.arctan2(dy_m, dx_m)
            sin_ang = np.sin(angle)
            cos_ang = np.cos(angle)
            
            rel_features.append([dist_log, sin_ang, cos_ang])
            
        relative_pos_dict[sid] = np.array(rel_features, dtype=np.float32)
    
    return neighbor_dict, relative_pos_dict


if __name__ == "__main__":
    result_nb, result_rel = get_station_neighbors('空氣品質監測站基本資料.csv', k=3)
    
    test_names = ["基隆", "苗栗", "馬祖"] 

    df_temp = pd.read_csv('空氣品質監測站基本資料.csv')
    
    print("-" * 80)
    print(f"{'站名':<4} | {'鄰居':<4} | {'真實距離(公尺)':<7} -> {'Log數值':<5} | {'Sin':<6} {'Cos':<6}")
    print("-" * 80)
    
    for name in test_names:
        try:
            s_row = df_temp[df_temp['sitename'] == name]
            if len(s_row) == 0:
                print(f"[{name}] 不在 CSV 中")
                continue
                
            sid = int(s_row['siteid'].values[0])
            
            if sid not in result_nb:
                print(f"[{name}] (ID:{sid}) 已被排除，無鄰居資料。")
                continue

            neighbors = result_nb[sid]
            rel_infos = result_rel[sid]
            
            print(f"[{name}]")
            for nb_id, rel in zip(neighbors, rel_infos):
                # 反推距離
                reconstructed_dist = 10**rel[0] - 1
                nb_name = df_temp[df_temp['siteid'] == nb_id]['sitename'].values[0]
                
                print(f"   -> {nb_name:<4} | {reconstructed_dist:>12.1f}m -> {rel[0]:.4f}   | {rel[1]:.2f}  {rel[2]:.2f}")
                
        except Exception as e:
            print(f"[{name}] Err: {e}")