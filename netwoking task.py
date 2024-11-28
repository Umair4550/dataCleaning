import os
from scapy.all import *
from scapy.layers.dns import DNS, DNSQR
from scapy.layers.inet import TCP, IP

# Function to list devices connected to the Wi-Fi network (via ARP or other methods)
def list_connected_devices():
    print("Listing devices connected to the network...\n")
    devices = os.popen("arp -a").readlines()
    connected_devices = {}

    for line in devices:
        if "dynamic" in line or "static" in line:
            parts = line.split()
            if len(parts) > 2:
                ip_address = parts[0]
                mac_address = parts[1]

                # Filter out multicast and broadcast IP addresses
                if not ip_address.startswith(('224', '255')):
                    connected_devices[ip_address] = mac_address

    print("\nConnected Devices:")
    for ip, mac in connected_devices.items():
        print(f"IP: {ip}, MAC: {mac}")
    print(f"\nTotal Connected Devices: {len(connected_devices)}")
    return connected_devices


# Function to identify the type of device based on the MAC address
def identify_device(mac_address):
    # Simplified vendor lookup (you can expand this with more known prefixes)
    if mac_address.startswith("00:1A:79") or mac_address.startswith("BC:92:6B"):
        return "Mobile"
    elif mac_address.startswith("00:19:D1") or mac_address.startswith("5C:F9:38"):
        return "Laptop"
    return "Unknown Device"


# Callback function to process packets and display search queries
def packet_callback(packet, device_activity):
    try:
        if packet.haslayer(IP) and packet.haslayer(TCP):
            src_ip = packet[IP].src

            # Check for HTTP Raw payload (search queries often appear in HTTP requests)
            if packet.haslayer(Raw):
                try:
                    payload = packet[Raw].load.decode(errors="ignore")
                    if "search" in payload.lower() or "q=" in payload.lower():  # Look for search queries
                        query = payload.split("q=")[-1].split(" ")[0]  # Extract query part from payload
                        print(f"{src_ip} query: {query}")
                        device_activity[src_ip]["search_activity"].append(f"query: {query}")
                except Exception as e:
                    print(f"Error decoding HTTP payload: {e}")

            # Check for DNS queries (e.g., lookups for domains related to searches)
            elif packet.haslayer(DNS) and packet[DNS].qd:
                domain = packet[DNSQR].qname.decode(errors="ignore").strip(".")
                if "google" in domain or "youtube" in domain or "bing" in domain:  # Search engine lookups
                    print(f"{src_ip} query: {domain}")
                    device_activity[src_ip]["search_activity"].append(f"query: {domain}")

    except Exception as e:
        print(f"Error processing packet: {e}")


# Main function to start sniffing and monitoring the network
def main():
    # List all connected devices (using ARP table)
    connected_devices = list_connected_devices()
    device_activity = {ip: {"mac": mac, "search_activity": []} for ip, mac in connected_devices.items()}

    # We will monitor all connected devices without manually specifying IPs
    print("\nMonitoring all devices connected to the network...")

    # Device list with identification (using MAC address for identification)
    print("\nDevice List with Type Identification:")
    for ip, details in device_activity.items():
        mac = details["mac"]
        device_type = identify_device(mac)
        print(f"IP: {ip}, MAC: {mac}, Type: {device_type}")

    # Network interface for sniffing (ensure your interface is in promiscuous mode)
    iface_name = "Wi-Fi"  # Replace this with the correct interface name (Wi-Fi or Ethernet)

    print(f"\nStarting packet sniffing on interface: {iface_name}")
    try:
        sniff(
            iface=iface_name,
            prn=lambda pkt: packet_callback(pkt, device_activity),
            store=False,
            filter="ip",  # Capture only IP packets
            promisc=True  # Promiscuous mode to capture all traffic
        )
    except Exception as e:
        print(f"Error sniffing packets: {e}")

    # Display search activity summary
    print("\nSearch Activity Summary:")
    for ip, details in device_activity.items():
        print(f"\nDevice IP: {ip}, MAC: {details['mac']}")
        if details["search_activity"]:
            print("  Search History:")
            for activity in details["search_activity"]:
                print(f"    {activity}")
        else:
            print("  No search activity recorded.")


if __name__ == "__main__":
    main()
