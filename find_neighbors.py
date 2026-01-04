import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

def get_station_neighbors(csv_path='空氣品質監測站基本資料.csv', k=3):
    # 1. 讀取資料
    df = pd.read_csv(csv_path)
    
    # --- 【新增修改】排除最後一個測站 ---
    # 取得最後一個測站的名字 (為了印出來提示)
    last_station_name = df.iloc[-1]['sitename']
    print(f"[Info] 已排除資料表中最後一個測站不參與計算: {last_station_name}")
    
    # 排除最後一列 (使用 slicing)
    df = df.iloc[:-1].reset_index(drop=True)
    # --------------------------------
    
    # 2. 準備 ID 與 座標
    site_ids = df['siteid'].astype(int).values
    site_names = df['sitename'].values
    name_to_id = dict(zip(site_names, site_ids))
    
    # 取得原始經緯度 (數值是度)
    coords_deg = df[['twd97lon', 'twd97lat']].values
    id_to_coord = dict(zip(site_ids, coords_deg))
    
    # --- 【修正步驟】計算距離矩陣時，先粗略轉成公尺 ---
    # 為了讓 distance_matrix 找鄰居找得準，我們先把座標轉成 "假公尺座標"
    # 台灣地區概略轉換係數：
    LON_TO_M = 102000.0  # 經度每度約 102 公里
    LAT_TO_M = 111000.0  # 緯度每度約 111 公里
    
    coords_meter = coords_deg.copy()
    coords_meter[:, 0] *= LON_TO_M
    coords_meter[:, 1] *= LAT_TO_M
    
    # 用公尺算距離矩陣 (這樣找最近鄰居才準確)
    dist_mtx = distance_matrix(coords_meter, coords_meter)
    
    neighbor_dict = {}
    relative_pos_dict = {} 
    
    # 定義區域輔助站 (維持不變)
    # 注意：如果被排除的那個站剛好在這裡面 (例如馬祖被排除了)，
    # 下面的程式碼會自動過濾掉 (if n in name_to_id)，所以不會報錯。
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
    
    # 3. 遍歷計算
    for i, sid in enumerate(site_ids):
        sid = int(sid)
        current_area = df.iloc[i]['areaname']
        
        # 目標站的原始經緯度
        target_lon, target_lat = coords_deg[i]
        
        # --- A. 找鄰居 ---
        nearest_idx = np.argsort(dist_mtx[i])[1:k+1]
        spatial_neighbor_ids = site_ids[nearest_idx].tolist()
        
        support_names = area_support_map.get(current_area, [])
        # 這裡會檢查 if n in name_to_id，所以被排除的站不會被加進來
        support_ids = [name_to_id[n] for n in support_names if n in name_to_id]
        
        combined_set = set(spatial_neighbor_ids + support_ids)
        if sid in combined_set: combined_set.remove(sid)
        final_neighbors = list(combined_set)
        neighbor_dict[sid] = final_neighbors
        
        # --- B. 計算相對位置 (轉公尺 -> Log) ---
        rel_features = []
        
        for nb_id in final_neighbors:
            nb_lon, nb_lat = id_to_coord[nb_id]
            
            # 1. 計算向量並轉為公尺 (Neighbor - Target)
            dx_deg = nb_lon - target_lon
            dy_deg = nb_lat - target_lat
            
            dx_m = dx_deg * LON_TO_M
            dy_m = dy_deg * LAT_TO_M
            
            # 2. 計算真實距離 (公尺)
            real_dist_m = np.sqrt(dx_m**2 + dy_m**2)
            
            # 3. Log Transform
            # 例如: 13km = 13000m -> log10(13001) ≈ 4.11
            dist_log = np.log10(real_dist_m + 1.0)
            
            # 4. 方位角 (使用轉成公尺後的 dx, dy 算角度才準，不然會變形)
            angle = np.arctan2(dy_m, dx_m)
            sin_ang = np.sin(angle)
            cos_ang = np.cos(angle)
            
            rel_features.append([dist_log, sin_ang, cos_ang])
            
        relative_pos_dict[sid] = np.array(rel_features, dtype=np.float32)
    
    return neighbor_dict, relative_pos_dict

# --- 測試 ---
if __name__ == "__main__":
    result_nb, result_rel = get_station_neighbors('空氣品質監測站基本資料.csv', k=3)
    # print(result_nb)
    
    test_names = ["基隆", "苗栗", "馬祖"] # 假設 "馬祖" 是最後一個，它應該會被排除

    df_temp = pd.read_csv('空氣品質監測站基本資料.csv')
    
    print("-" * 80)
    print(f"{'站名':<4} | {'鄰居':<4} | {'真實距離(公尺)':<7} -> {'Log數值':<5} | {'Sin':<6} {'Cos':<6}")
    print("-" * 80)
    
    for name in test_names:
        try:
            # 嘗試取得 ID
            s_row = df_temp[df_temp['sitename'] == name]
            if len(s_row) == 0:
                print(f"[{name}] 不在 CSV 中")
                continue
                
            sid = int(s_row['siteid'].values[0])
            
            # 檢查這個站是否有被計算 (若被排除，這裡會報 Key Error)
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