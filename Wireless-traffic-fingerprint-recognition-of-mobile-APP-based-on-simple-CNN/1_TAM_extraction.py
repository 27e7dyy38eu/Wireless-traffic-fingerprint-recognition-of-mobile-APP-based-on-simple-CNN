from scapy.all import rdpcap, PcapReader
import csv
import os


def analyze_pcap(pcap_file, target_mac, T, S):
    """
    读取 pcap 文件并统计从第一个数据包开始的 T 秒内每 S 秒的上行帧和下行帧数量。

    :param pcap_file: pcap 文件路径
    :param target_mac: 指定的目标 MAC 地址
    :param T: 总时间（秒）
    :param S: 时间间隔（秒）
    :return: 一个二维列表，包含每 S 秒的上行帧和下行帧数量
    """
    
    try:
        with PcapReader(pcap_file) as pcap_reader:
            packets = iter(pcap_reader)
            first_packet = next(packets)
            start_time = first_packet.time

            intervals = int(T / S)  
            result = [[0, 0] for _ in range(intervals)]  

            timestamp = first_packet.time
            if timestamp <= start_time + T and 'Dot11' in first_packet:
                relative_time = timestamp - start_time
                interval_index = int(relative_time // S)
                if interval_index < intervals and first_packet.type == 2:  
                    src_mac = first_packet.addr2  
                    dst_mac = first_packet.addr1  
                    if src_mac == target_mac:
                        result[interval_index][0] += 1  
                    elif dst_mac == target_mac:
                        result[interval_index][1] += 1  

            
            for packet in packets:
                timestamp = packet.time
                if timestamp > start_time + T:
                    break

                relative_time = timestamp - start_time
                interval_index = int(relative_time // S)
                if interval_index >= intervals:
                    continue

                if 'Dot11' in packet and packet.type == 2:  
                    src_mac = packet.addr2  
                    dst_mac = packet.addr1  
                    if src_mac == target_mac:
                        result[interval_index][0] += 1  
                    elif dst_mac == target_mac:
                        result[interval_index][1] += 1  

        return result
    except Exception as e:
        print(f"处理 {pcap_file} 时出错: {e}")
        return []


def write_to_csv(result, csv_file):
    """
    将统计结果写入 CSV 文件。

    :param result: 二维列表，包含每 S 秒的上行帧和下行帧数量
    :param csv_file: 输出的 CSV 文件路径
    """
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["上行帧数量", "下行帧数量"])
        for up_count, down_count in result:
            writer.writerow([up_count, down_count])

    print(f"结果已成功写入 {csv_file}")


def process_folder(folder_path, target_mac, T, S, output_folder):
    """
    处理指定文件夹中的所有 .pcap 文件，并为每个文件生成对应的 CSV 文件。

    :param folder_path: 包含 .pcap 文件的文件夹路径
    :param target_mac: 指定的目标 MAC 地址
    :param T: 总时间（秒）
    :param S: 时间间隔（秒）
    :param output_folder: 输出 CSV 文件的文件夹路径
    """
    
    os.makedirs(output_folder, exist_ok=True)

    
    pcap_files = [f for f in os.listdir(folder_path) if f.endswith('.pcap')]
    pcap_files.sort()  

    
    for i, pcap_file in enumerate(pcap_files, start=1):
        pcap_file_path = os.path.join(folder_path, pcap_file)
        print(f"正在处理文件: {pcap_file_path}")

        
        result = analyze_pcap(pcap_file_path, target_mac, T, S)

        
        csv_file_name = f"{i}.csv"
        csv_file_path = os.path.join(output_folder, csv_file_name)

        
        write_to_csv(result, csv_file_path)



if __name__ == "__main__":
    
    folder_path = "wireless_pcap/9抖音"
    target_mac = "aa:bb:cc:dd:ee:ff"  
    T = 2  
    S = 0.1  
    output_folder = "TAM/9抖音"
    process_folder(folder_path, target_mac, T, S, output_folder)
