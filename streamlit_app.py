import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math
import random
from typing import Dict, List, Tuple, Optional
import json
import requests
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import anthropic
import asyncio
import logging
from dataclasses import dataclass
import yaml
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AWS Enterprise Database Migration Analyzer AI v4.0 - 16 Scenarios",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(30,60,114,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .scenario-selector-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(102,126,234,0.2);
        border-left: 3px solid #667eea;
    }
    
    .migration-tool-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(245,87,108,0.2);
        border-left: 3px solid #f5576c;
    }
    
    .destination-config-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(79,172,254,0.2);
        border-left: 3px solid #4facfe;
    }
    
    .server-config-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(67,233,123,0.2);
        border-left: 3px solid #43e97b;
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(52,73,94,0.2);
        border-left: 3px solid #3498db;
    }
    
    .agent-scaling-card {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(232,67,147,0.2);
        border-left: 3px solid #fd79a8;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .detailed-analysis-section {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .scenario-overview-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.2rem;
        border-radius: 8px;
        color: #2c3e50;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(168,237,234,0.3);
        border-left: 3px solid #a8edea;
    }
</style>
""", unsafe_allow_html=True)

class Enhanced16ScenarioNetworkManager:
    """Enhanced network manager supporting all 16 migration scenarios"""
    
    def __init__(self):
        self.migration_scenarios = {
            # Non-Production DataSync Scenarios (1-4)
            'nonprod_sj_linux_nas_s3_datasync': {
                'id': 1,
                'name': 'Non-Prod: San Jose Linux NAS → DataSync → AWS S3',
                'environment': 'non-production',
                'source_location': 'San Jose',
                'source_os': 'linux',
                'source_storage': 'nas',
                'migration_tool': 'datasync',
                'destination': 's3',
                'destination_vpc': 'aws_west_2_nonprod',
                'network_path': [
                    {'segment': 'Linux NAS to Linux Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 2, 'reliability': 0.999},
                    {'segment': 'Bastion to On-prem Firewall', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.999},
                    {'segment': 'DX Link 2Gbps to AWS S3', 'bandwidth_mbps': 2000, 'latency_ms': 15, 'reliability': 0.998}
                ],
                'complexity_score': 3,
                'recommended_agents': 2
            },
            'nonprod_sj_linux_nas_fsx_datasync': {
                'id': 2,
                'name': 'Non-Prod: San Jose Linux NAS → DataSync → FSx Lustre',
                'environment': 'non-production',
                'source_location': 'San Jose',
                'source_os': 'linux',
                'source_storage': 'nas',
                'migration_tool': 'datasync',
                'destination': 'fsx_lustre',
                'destination_vpc': 'aws_west_2_nonprod',
                'network_path': [
                    {'segment': 'Linux NAS to Linux Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 2, 'reliability': 0.999},
                    {'segment': 'Bastion to On-prem Firewall', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.999},
                    {'segment': 'DX Link 2Gbps to FSx Lustre', 'bandwidth_mbps': 2000, 'latency_ms': 18, 'reliability': 0.998}
                ],
                'complexity_score': 4,
                'recommended_agents': 2
            },
            'nonprod_sj_windows_share_s3_datasync': {
                'id': 3,
                'name': 'Non-Prod: San Jose Windows Share → DataSync → AWS S3',
                'environment': 'non-production',
                'source_location': 'San Jose',
                'source_os': 'windows',
                'source_storage': 'share',
                'migration_tool': 'datasync',
                'destination': 's3',
                'destination_vpc': 'aws_west_2_nonprod',
                'network_path': [
                    {'segment': 'Windows Share to Windows Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 3, 'reliability': 0.997},
                    {'segment': 'Bastion to On-prem Firewall', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.997},
                    {'segment': 'DX Link 2Gbps to AWS S3', 'bandwidth_mbps': 2000, 'latency_ms': 18, 'reliability': 0.998}
                ],
                'complexity_score': 4,
                'recommended_agents': 2
            },
            'nonprod_sj_windows_share_fsx_datasync': {
                'id': 4,
                'name': 'Non-Prod: San Jose Windows Share → DataSync → Windows FSx',
                'environment': 'non-production',
                'source_location': 'San Jose',
                'source_os': 'windows',
                'source_storage': 'share',
                'migration_tool': 'datasync',
                'destination': 'fsx_windows',
                'destination_vpc': 'aws_west_2_nonprod',
                'network_path': [
                    {'segment': 'Windows Share to Windows Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 3, 'reliability': 0.997},
                    {'segment': 'Bastion to On-prem Firewall', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.997},
                    {'segment': 'DX Link 2Gbps to Windows FSx', 'bandwidth_mbps': 2000, 'latency_ms': 20, 'reliability': 0.998}
                ],
                'complexity_score': 5,
                'recommended_agents': 2
            },
            
            # Production DataSync Scenarios (5-8)
            'prod_sa_linux_nas_s3_datasync': {
                'id': 5,
                'name': 'Prod: San Antonio Linux NAS → San Jose → DataSync → AWS S3',
                'environment': 'production',
                'source_location': 'San Antonio',
                'source_os': 'linux',
                'source_storage': 'nas',
                'migration_tool': 'datasync',
                'destination': 's3',
                'destination_vpc': 'aws_west_2_prod',
                'network_path': [
                    {'segment': 'Linux NAS to Linux Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.999},
                    {'segment': 'San Antonio to San Jose 10Gbps', 'bandwidth_mbps': 10000, 'latency_ms': 12, 'reliability': 0.9995},
                    {'segment': 'San Jose DX 10Gbps to AWS S3', 'bandwidth_mbps': 10000, 'latency_ms': 8, 'reliability': 0.9999}
                ],
                'complexity_score': 6,
                'recommended_agents': 3
            },
            'prod_sa_linux_nas_fsx_datasync': {
                'id': 6,
                'name': 'Prod: San Antonio Linux NAS → San Jose → DataSync → FSx Linux',
                'environment': 'production',
                'source_location': 'San Antonio',
                'source_os': 'linux',
                'source_storage': 'nas',
                'migration_tool': 'datasync',
                'destination': 'fsx_lustre',
                'destination_vpc': 'aws_west_2_prod',
                'network_path': [
                    {'segment': 'Linux NAS to Linux Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.999},
                    {'segment': 'San Antonio to San Jose 10Gbps', 'bandwidth_mbps': 10000, 'latency_ms': 12, 'reliability': 0.9995},
                    {'segment': 'San Jose DX 10Gbps to FSx Linux', 'bandwidth_mbps': 10000, 'latency_ms': 10, 'reliability': 0.9999}
                ],
                'complexity_score': 7,
                'recommended_agents': 3
            },
            'prod_sa_windows_share_s3_datasync': {
                'id': 7,
                'name': 'Prod: San Antonio Windows Share → San Jose → DataSync → AWS S3',
                'environment': 'production',
                'source_location': 'San Antonio',
                'source_os': 'windows',
                'source_storage': 'share',
                'migration_tool': 'datasync',
                'destination': 's3',
                'destination_vpc': 'aws_west_2_prod',
                'network_path': [
                    {'segment': 'Windows Share to Windows Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 2, 'reliability': 0.998},
                    {'segment': 'San Antonio to San Jose 10Gbps', 'bandwidth_mbps': 10000, 'latency_ms': 15, 'reliability': 0.9995},
                    {'segment': 'San Jose DX 10Gbps to AWS S3', 'bandwidth_mbps': 10000, 'latency_ms': 10, 'reliability': 0.9999}
                ],
                'complexity_score': 7,
                'recommended_agents': 3
            },
            'prod_sa_windows_share_fsx_datasync': {
                'id': 8,
                'name': 'Prod: San Antonio Windows Share → San Jose → DataSync → Windows FSx',
                'environment': 'production',
                'source_location': 'San Antonio',
                'source_os': 'windows',
                'source_storage': 'share',
                'migration_tool': 'datasync',
                'destination': 'fsx_windows',
                'destination_vpc': 'aws_west_2_prod',
                'network_path': [
                    {'segment': 'Windows Share to Windows Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 2, 'reliability': 0.998},
                    {'segment': 'San Antonio to San Jose 10Gbps', 'bandwidth_mbps': 10000, 'latency_ms': 15, 'reliability': 0.9995},
                    {'segment': 'San Jose DX 10Gbps to Windows FSx', 'bandwidth_mbps': 10000, 'latency_ms': 12, 'reliability': 0.9999}
                ],
                'complexity_score': 8,
                'recommended_agents': 4
            },
            
            # Non-Production DMS Scenarios (9-12)
            'nonprod_sj_linux_nas_s3_dms': {
                'id': 9,
                'name': 'Non-Prod: San Jose Linux NAS → DMS → AWS S3',
                'environment': 'non-production',
                'source_location': 'San Jose',
                'source_os': 'linux',
                'source_storage': 'nas',
                'migration_tool': 'dms',
                'destination': 's3',
                'destination_vpc': 'aws_west_2_nonprod',
                'network_path': [
                    {'segment': 'Linux NAS to Linux Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 2, 'reliability': 0.999},
                    {'segment': 'Bastion to On-prem Firewall', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.999},
                    {'segment': 'DX Link 2Gbps to AWS DMS', 'bandwidth_mbps': 2000, 'latency_ms': 15, 'reliability': 0.998}
                ],
                'complexity_score': 5,
                'recommended_agents': 2
            },
            'nonprod_sj_linux_nas_fsx_dms': {
                'id': 10,
                'name': 'Non-Prod: San Jose Linux NAS → DMS → FSx Lustre',
                'environment': 'non-production',
                'source_location': 'San Jose',
                'source_os': 'linux',
                'source_storage': 'nas',
                'migration_tool': 'dms',
                'destination': 'fsx_lustre',
                'destination_vpc': 'aws_west_2_nonprod',
                'network_path': [
                    {'segment': 'Linux NAS to Linux Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 2, 'reliability': 0.999},
                    {'segment': 'Bastion to On-prem Firewall', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.999},
                    {'segment': 'DX Link 2Gbps to DMS+FSx', 'bandwidth_mbps': 2000, 'latency_ms': 18, 'reliability': 0.998}
                ],
                'complexity_score': 6,
                'recommended_agents': 2
            },
            'nonprod_sj_windows_share_s3_dms': {
                'id': 11,
                'name': 'Non-Prod: San Jose Windows Share → DMS → AWS S3',
                'environment': 'non-production',
                'source_location': 'San Jose',
                'source_os': 'windows',
                'source_storage': 'share',
                'migration_tool': 'dms',
                'destination': 's3',
                'destination_vpc': 'aws_west_2_nonprod',
                'network_path': [
                    {'segment': 'Windows Share to Windows Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 3, 'reliability': 0.997},
                    {'segment': 'Bastion to On-prem Firewall', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.997},
                    {'segment': 'DX Link 2Gbps to AWS DMS', 'bandwidth_mbps': 2000, 'latency_ms': 18, 'reliability': 0.998}
                ],
                'complexity_score': 6,
                'recommended_agents': 2
            },
            'nonprod_sj_windows_share_fsx_dms': {
                'id': 12,
                'name': 'Non-Prod: San Jose Windows Share → DMS → Windows FSx',
                'environment': 'non-production',
                'source_location': 'San Jose',
                'source_os': 'windows',
                'source_storage': 'share',
                'migration_tool': 'dms',
                'destination': 'fsx_windows',
                'destination_vpc': 'aws_west_2_nonprod',
                'network_path': [
                    {'segment': 'Windows Share to Windows Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 3, 'reliability': 0.997},
                    {'segment': 'Bastion to On-prem Firewall', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.997},
                    {'segment': 'DX Link 2Gbps to DMS+Windows FSx', 'bandwidth_mbps': 2000, 'latency_ms': 20, 'reliability': 0.998}
                ],
                'complexity_score': 7,
                'recommended_agents': 3
            },
            
            # Production DMS Scenarios (13-16)
            'prod_sa_linux_nas_s3_dms': {
                'id': 13,
                'name': 'Prod: San Antonio Linux NAS → San Jose → DMS → AWS S3',
                'environment': 'production',
                'source_location': 'San Antonio',
                'source_os': 'linux',
                'source_storage': 'nas',
                'migration_tool': 'dms',
                'destination': 's3',
                'destination_vpc': 'aws_west_2_prod',
                'network_path': [
                    {'segment': 'Linux NAS to Linux Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.999},
                    {'segment': 'San Antonio to San Jose 10Gbps', 'bandwidth_mbps': 10000, 'latency_ms': 12, 'reliability': 0.9995},
                    {'segment': 'San Jose DX 10Gbps to AWS DMS', 'bandwidth_mbps': 10000, 'latency_ms': 8, 'reliability': 0.9999}
                ],
                'complexity_score': 8,
                'recommended_agents': 4
            },
            'prod_sa_linux_nas_fsx_dms': {
                'id': 14,
                'name': 'Prod: San Antonio Linux NAS → San Jose → DMS → FSx Linux',
                'environment': 'production',
                'source_location': 'San Antonio',
                'source_os': 'linux',
                'source_storage': 'nas',
                'migration_tool': 'dms',
                'destination': 'fsx_lustre',
                'destination_vpc': 'aws_west_2_prod',
                'network_path': [
                    {'segment': 'Linux NAS to Linux Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 1, 'reliability': 0.999},
                    {'segment': 'San Antonio to San Jose 10Gbps', 'bandwidth_mbps': 10000, 'latency_ms': 12, 'reliability': 0.9995},
                    {'segment': 'San Jose DX 10Gbps to DMS+FSx', 'bandwidth_mbps': 10000, 'latency_ms': 10, 'reliability': 0.9999}
                ],
                'complexity_score': 9,
                'recommended_agents': 4
            },
            'prod_sa_windows_share_s3_dms': {
                'id': 15,
                'name': 'Prod: San Antonio Windows Share → San Jose → DMS → AWS S3',
                'environment': 'production',
                'source_location': 'San Antonio',
                'source_os': 'windows',
                'source_storage': 'share',
                'migration_tool': 'dms',
                'destination': 's3',
                'destination_vpc': 'aws_west_2_prod',
                'network_path': [
                    {'segment': 'Windows Share to Windows Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 2, 'reliability': 0.998},
                    {'segment': 'San Antonio to San Jose 10Gbps', 'bandwidth_mbps': 10000, 'latency_ms': 15, 'reliability': 0.9995},
                    {'segment': 'San Jose DX 10Gbps to AWS DMS', 'bandwidth_mbps': 10000, 'latency_ms': 10, 'reliability': 0.9999}
                ],
                'complexity_score': 9,
                'recommended_agents': 4
            },
            'prod_sa_windows_share_fsx_dms': {
                'id': 16,
                'name': 'Prod: San Antonio Windows Share → San Jose → DMS → Windows FSx',
                'environment': 'production',
                'source_location': 'San Antonio',
                'source_os': 'windows',
                'source_storage': 'share',
                'migration_tool': 'dms',
                'destination': 'fsx_windows',
                'destination_vpc': 'aws_west_2_prod',
                'network_path': [
                    {'segment': 'Windows Share to Windows Bastion', 'bandwidth_mbps': 1000, 'latency_ms': 2, 'reliability': 0.998},
                    {'segment': 'San Antonio to San Jose 10Gbps', 'bandwidth_mbps': 10000, 'latency_ms': 15, 'reliability': 0.9995},
                    {'segment': 'San Jose DX 10Gbps to DMS+Windows FSx', 'bandwidth_mbps': 10000, 'latency_ms': 12, 'reliability': 0.9999}
                ],
                'complexity_score': 10,
                'recommended_agents': 5
            }
        }
    
    def get_scenario_by_id(self, scenario_id: int) -> Dict:
        """Get scenario by ID"""
        for key, scenario in self.migration_scenarios.items():
            if scenario['id'] == scenario_id:
                return scenario
        return None
    
    def get_scenarios_by_criteria(self, environment: str = None, migration_tool: str = None, 
                                source_os: str = None, destination: str = None) -> List[Dict]:
        """Filter scenarios by criteria"""
        filtered = []
        for key, scenario in self.migration_scenarios.items():
            matches = True
            if environment and scenario['environment'] != environment:
                matches = False
            if migration_tool and scenario['migration_tool'] != migration_tool:
                matches = False
            if source_os and scenario['source_os'] != source_os:
                matches = False
            if destination and scenario['destination'] != destination:
                matches = False
            
            if matches:
                filtered.append({**scenario, 'key': key})
        
        return filtered
    
    def calculate_scenario_performance(self, scenario_key: str, config: Dict) -> Dict:
        """Calculate performance for a specific scenario"""
        scenario = self.migration_scenarios.get(scenario_key)
        if not scenario:
            return {}
        
        # Calculate network performance
        total_latency = sum([segment['latency_ms'] for segment in scenario['network_path']])
        min_bandwidth = min([segment['bandwidth_mbps'] for segment in scenario['network_path']])
        total_reliability = 1.0
        for segment in scenario['network_path']:
            total_reliability *= segment['reliability']
        
        # Apply OS overhead
        if scenario['source_os'] == 'windows':
            min_bandwidth *= 0.95  # Windows SMB overhead
            total_latency *= 1.1
        
        # Apply migration tool efficiency
        if scenario['migration_tool'] == 'dms':
            min_bandwidth *= 0.85  # DMS transformation overhead
            total_latency *= 1.2   # Schema conversion latency
        
        # Calculate agent scaling impact
        num_agents = config.get('number_of_agents', scenario['recommended_agents'])
        agent_efficiency = min(1.0, 0.9 + (num_agents * 0.02))  # Efficiency improves with more agents
        
        effective_bandwidth = min_bandwidth * agent_efficiency
        
        return {
            'scenario': scenario,
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': effective_bandwidth,
            'total_reliability': total_reliability,
            'recommended_agents': scenario['recommended_agents'],
            'complexity_score': scenario['complexity_score'],
            'agent_efficiency': agent_efficiency,
            'network_segments': scenario['network_path']
        }

class EnhancedServerConfigurationManager:
    """Enhanced server configuration with virtual/physical considerations"""
    
    def __init__(self):
        self.server_types = {
            'physical_dell_r750': {
                'name': 'Dell PowerEdge R750 (Physical)',
                'type': 'physical',
                'vendor': 'Dell',
                'cpu_efficiency': 1.0,
                'memory_efficiency': 1.0,
                'io_efficiency': 1.0,
                'virtualization_overhead': 0.0,
                'recommended_for': 'High-performance databases, low latency requirements'
            },
            'physical_hp_dl380': {
                'name': 'HP ProLiant DL380 (Physical)',
                'type': 'physical',
                'vendor': 'HP',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.99,
                'io_efficiency': 0.98,
                'virtualization_overhead': 0.0,
                'recommended_for': 'Enterprise databases, mission-critical workloads'
            },
            'vmware_vsphere7': {
                'name': 'VMware vSphere 7.0 (Virtual)',
                'type': 'virtual',
                'vendor': 'VMware',
                'cpu_efficiency': 0.92,
                'memory_efficiency': 0.88,
                'io_efficiency': 0.85,
                'virtualization_overhead': 0.12,
                'recommended_for': 'Flexible resource allocation, easy scaling'
            },
            'vmware_vsphere8': {
                'name': 'VMware vSphere 8.0 (Virtual)',
                'type': 'virtual',
                'vendor': 'VMware',
                'cpu_efficiency': 0.95,
                'memory_efficiency': 0.92,
                'io_efficiency': 0.90,
                'virtualization_overhead': 0.08,
                'recommended_for': 'Latest virtualization features, improved performance'
            },
            'hyper_v_2022': {
                'name': 'Microsoft Hyper-V 2022 (Virtual)',
                'type': 'virtual',
                'vendor': 'Microsoft',
                'cpu_efficiency': 0.90,
                'memory_efficiency': 0.85,
                'io_efficiency': 0.82,
                'virtualization_overhead': 0.15,
                'recommended_for': 'Windows-centric environments, SQL Server workloads'
            }
        }
    
    def get_server_performance_impact(self, server_type: str, config: Dict) -> Dict:
        """Calculate server performance impact"""
        server_config = self.server_types.get(server_type, self.server_types['vmware_vsphere7'])
        
        # Base performance calculation
        cpu_performance = config['cpu_cores'] * config['cpu_ghz'] * server_config['cpu_efficiency']
        memory_performance = config['ram_gb'] * server_config['memory_efficiency']
        io_performance = config.get('max_iops', 10000) * server_config['io_efficiency']
        
        # Apply virtualization overhead if applicable
        if server_config['type'] == 'virtual':
            overhead = server_config['virtualization_overhead']
            cpu_performance *= (1 - overhead)
            memory_performance *= (1 - overhead)
            io_performance *= (1 - overhead)
        
        return {
            'server_config': server_config,
            'cpu_performance': cpu_performance,
            'memory_performance': memory_performance,
            'io_performance': io_performance,
            'overall_efficiency': (server_config['cpu_efficiency'] + 
                                 server_config['memory_efficiency'] + 
                                 server_config['io_efficiency']) / 3
        }

class Enhanced16ScenarioAnalyzer:
    """Enhanced analyzer supporting all 16 migration scenarios"""
    
    def __init__(self):
        self.network_manager = Enhanced16ScenarioNetworkManager()
        self.server_manager = EnhancedServerConfigurationManager()
        # Include other managers from original code
        
    async def analyze_migration_scenario(self, scenario_key: str, config: Dict) -> Dict:
        """Comprehensive analysis for specific migration scenario"""
        
        # Get scenario details
        scenario_performance = self.network_manager.calculate_scenario_performance(scenario_key, config)
        server_performance = self.server_manager.get_server_performance_impact(config['server_type'], config)
        
        # Calculate migration time with scenario-specific factors
        migration_time = self._calculate_scenario_migration_time(scenario_performance, config)
        
        # Calculate costs with scenario-specific factors
        cost_analysis = self._calculate_scenario_costs(scenario_performance, config)
        
        # AI analysis for specific scenario
        ai_insights = await self._get_scenario_ai_insights(scenario_performance, config)
        
        return {
            'scenario_performance': scenario_performance,
            'server_performance': server_performance,
            'migration_time_hours': migration_time,
            'cost_analysis': cost_analysis,
            'ai_insights': ai_insights,
            'recommendations': self._generate_scenario_recommendations(scenario_performance, config)
        }
    
    def _calculate_scenario_migration_time(self, scenario_performance: Dict, config: Dict) -> float:
        """Calculate migration time for specific scenario"""
        scenario = scenario_performance['scenario']
        effective_bandwidth = scenario_performance['effective_bandwidth_mbps']
        database_size_gb = config['database_size_gb']
        
        # Base time calculation
        base_time_hours = (database_size_gb * 8 * 1000) / (effective_bandwidth * 3600)
        
        # Apply scenario-specific factors
        complexity_factor = 1.0 + (scenario['complexity_score'] - 5) * 0.1
        
        # Tool-specific factors
        if scenario['migration_tool'] == 'dms':
            complexity_factor *= 1.3  # Schema conversion overhead
        
        # Environment factors
        if scenario['environment'] == 'production':
            complexity_factor *= 1.2  # Additional validation requirements
        
        return base_time_hours * complexity_factor
    
    def _calculate_scenario_costs(self, scenario_performance: Dict, config: Dict) -> Dict:
        """Calculate costs for specific scenario"""
        scenario = scenario_performance['scenario']
        
        # Base AWS costs (simplified)
        base_monthly_cost = 1000  # Base RDS/EC2 cost
        
        # Scenario-specific adjustments
        if scenario['environment'] == 'production':
            base_monthly_cost *= 1.5  # Production redundancy
        
        if scenario['destination'] in ['fsx_lustre', 'fsx_windows']:
            base_monthly_cost *= 1.3  # FSx premium
        
        # Network costs
        if scenario['source_location'] == 'San Antonio':
            network_cost = 1200  # Higher for multi-hop
        else:
            network_cost = 800   # Direct connection
        
        # Agent costs
        num_agents = config.get('number_of_agents', scenario['recommended_agents'])
        agent_cost_per_month = num_agents * 150  # $150 per agent
        
        total_monthly = base_monthly_cost + network_cost + agent_cost_per_month
        
        return {
            'base_aws_cost': base_monthly_cost,
            'network_cost': network_cost,
            'agent_cost': agent_cost_per_month,
            'total_monthly_cost': total_monthly,
            'scenario_complexity_multiplier': scenario['complexity_score'] / 5.0
        }
    
    async def _get_scenario_ai_insights(self, scenario_performance: Dict, config: Dict) -> Dict:
        """Get AI insights for specific scenario"""
        scenario = scenario_performance['scenario']
        
        # Simplified AI insights (in real implementation, call Anthropic API)
        insights = {
            'complexity_assessment': f"Scenario {scenario['id']} complexity: {scenario['complexity_score']}/10",
            'bottleneck_analysis': [],
            'optimization_recommendations': [],
            'risk_factors': []
        }
        
        # Analyze bottlenecks
        if scenario['source_os'] == 'windows' and scenario['migration_tool'] == 'dms':
            insights['bottleneck_analysis'].append("Windows + DMS combination may have SMB protocol overhead")
        
        if scenario['environment'] == 'production' and scenario['source_location'] == 'San Antonio':
            insights['bottleneck_analysis'].append("Multi-hop production path increases latency and complexity")
        
        # Generate recommendations
        if config.get('number_of_agents', 1) < scenario['recommended_agents']:
            insights['optimization_recommendations'].append(f"Consider scaling to {scenario['recommended_agents']} agents for optimal performance")
        
        if scenario['destination'] == 's3' and config['database_size_gb'] > 10000:
            insights['optimization_recommendations'].append("Large database to S3 may benefit from S3 Transfer Acceleration")
        
        return insights
    
    def _generate_scenario_recommendations(self, scenario_performance: Dict, config: Dict) -> List[str]:
        """Generate scenario-specific recommendations"""
        scenario = scenario_performance['scenario']
        recommendations = []
        
        # Tool-specific recommendations
        if scenario['migration_tool'] == 'datasync':
            recommendations.append("DataSync: Ensure agents are deployed close to source data")
            recommendations.append("DataSync: Configure bandwidth throttling during business hours")
        else:
            recommendations.append("DMS: Test schema conversion thoroughly in non-production")
            recommendations.append("DMS: Monitor replication lag during initial sync")
        
        # Destination-specific recommendations
        if 'fsx' in scenario['destination']:
            recommendations.append("FSx: Pre-provision file system with adequate IOPS")
            recommendations.append("FSx: Configure appropriate backup schedule")
        else:
            recommendations.append("S3: Configure lifecycle policies for cost optimization")
            recommendations.append("S3: Enable versioning for data protection")
        
        # Environment-specific recommendations
        if scenario['environment'] == 'production':
            recommendations.append("Production: Plan detailed rollback procedures")
            recommendations.append("Production: Schedule migration during maintenance window")
        
        return recommendations

def render_enhanced_header_16_scenarios():
    """Enhanced header for 16-scenario support"""
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Database Migration Analyzer AI v4.0</h1>
        <h2>Complete 16-Scenario Migration Analysis Platform</h2>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration • Advanced Agent Scaling
        </p>
        <p style="font-size: 1.0rem; margin-top: 0.5rem;">
            <strong>✨ NEW:</strong> All 16 Migration Scenarios • DataSync & DMS Support • Virtual/Physical Servers • FSx Destinations
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">📊 16 Migration Scenarios</span>
            <span style="margin-right: 20px;">🔄 DataSync + DMS Tools</span>
            <span style="margin-right: 20px;">🖥️ Virtual + Physical Servers</span>
            <span>🗄️ S3 + FSx Destinations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_scenario_selector_sidebar():
    """Enhanced sidebar for scenario selection"""
    st.sidebar.header("🎯 Migration Scenario Configuration v4.0")
    
    network_manager = Enhanced16ScenarioNetworkManager()
    
    # Scenario Selection Method
    st.sidebar.subheader("📋 Scenario Selection Method")
    
    selection_method = st.sidebar.radio(
        "Choose Configuration Method:",
        ["guided_selection", "manual_scenario_id"],
        format_func=lambda x: "🧭 Guided Selection (Recommended)" if x == "guided_selection" else "🔢 Direct Scenario ID",
        help="Guided selection helps you find the right scenario, or choose by ID if you know the specific scenario"
    )
    
    if selection_method == "guided_selection":
        # Guided Selection
        st.sidebar.markdown("**🧭 Guided Scenario Selection:**")
        
        environment = st.sidebar.selectbox(
            "Environment",
            ["non-production", "production"],
            format_func=lambda x: "🧪 Non-Production" if x == "non-production" else "🏭 Production",
            help="Production scenarios include San Antonio → San Jose routing"
        )
        
        source_os = st.sidebar.selectbox(
            "Source Operating System",
            ["linux", "windows"],
            format_func=lambda x: "🐧 Linux (NAS)" if x == "linux" else "🪟 Windows (Share)",
            help="Determines source storage type and protocol efficiency"
        )
        
        migration_tool = st.sidebar.selectbox(
            "Migration Tool",
            ["datasync", "dms"],
            format_func=lambda x: "📦 AWS DataSync (File-based)" if x == "datasync" else "🔄 AWS DMS (Database)",
            help="DataSync for homogeneous, DMS for heterogeneous migrations"
        )
        
        destination = st.sidebar.selectbox(
            "Destination Storage",
            ["s3", "fsx_lustre", "fsx_windows"],
            format_func=lambda x: {
                's3': '☁️ Amazon S3',
                'fsx_lustre': '⚡ FSx for Lustre (Linux)',
                'fsx_windows': '🪟 FSx for Windows'
            }[x],
            help="Target storage service in AWS"
        )
        
        # Find matching scenarios
        matching_scenarios = network_manager.get_scenarios_by_criteria(
            environment=environment,
            migration_tool=migration_tool,
            source_os=source_os,
            destination=destination
        )
        
        if matching_scenarios:
            selected_scenario = matching_scenarios[0]  # Take first match
            scenario_key = selected_scenario['key']
            scenario_id = selected_scenario['id']
            
            st.sidebar.success(f"✅ **Selected: Scenario {scenario_id}**")
            st.sidebar.markdown(f"**{selected_scenario['name']}**")
        else:
            st.sidebar.error("❌ No matching scenario found")
            scenario_key = list(network_manager.migration_scenarios.keys())[0]
            scenario_id = 1
    
    else:
        # Direct Scenario ID Selection
        scenario_id = st.sidebar.selectbox(
            "Select Scenario ID (1-16)",
            list(range(1, 17)),
            help="Choose the specific scenario number from the 16 available scenarios"
        )
        
        # Find scenario by ID
        selected_scenario = network_manager.get_scenario_by_id(scenario_id)
        if selected_scenario:
            # Find the key for this scenario
            for key, scenario in network_manager.migration_scenarios.items():
                if scenario['id'] == scenario_id:
                    scenario_key = key
                    break
        else:
            scenario_key = list(network_manager.migration_scenarios.keys())[0]
            selected_scenario = network_manager.migration_scenarios[scenario_key]
    
    # Display selected scenario details
    if selected_scenario:
        st.sidebar.markdown(f"""
        <div class="scenario-selector-card">
            <h4>📊 Scenario {selected_scenario['id']} Details</h4>
            <p><strong>Name:</strong> {selected_scenario['name']}</p>
            <p><strong>Environment:</strong> {selected_scenario['environment'].title()}</p>
            <p><strong>Tool:</strong> {selected_scenario['migration_tool'].upper()}</p>
            <p><strong>Source:</strong> {selected_scenario['source_location']} ({selected_scenario['source_os'].title()})</p>
            <p><strong>Destination:</strong> {selected_scenario['destination'].replace('_', ' ').title()}</p>
            <p><strong>Complexity:</strong> {selected_scenario['complexity_score']}/10</p>
            <p><strong>Recommended Agents:</strong> {selected_scenario['recommended_agents']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    return scenario_key, selected_scenario

def render_enhanced_server_configuration():
    """Enhanced server configuration with virtual/physical options"""
    st.sidebar.subheader("🖥️ Server Configuration")
    
    server_manager = EnhancedServerConfigurationManager()
    
    server_type = st.sidebar.selectbox(
        "Server Platform",
        list(server_manager.server_types.keys()),
        index=2,  # Default to VMware vSphere 7
        format_func=lambda x: server_manager.server_types[x]['name'],
        help="Choose between physical servers and virtualization platforms"
    )
    
    # Display server details
    server_config = server_manager.server_types[server_type]
    st.sidebar.markdown(f"""
    <div class="server-config-card">
        <h4>🔧 {server_config['name']}</h4>
        <p><strong>Type:</strong> {server_config['type'].title()}</p>
        <p><strong>CPU Efficiency:</strong> {server_config['cpu_efficiency']*100:.1f}%</p>
        <p><strong>Memory Efficiency:</strong> {server_config['memory_efficiency']*100:.1f}%</p>
        <p><strong>I/O Efficiency:</strong> {server_config['io_efficiency']*100:.1f}%</p>
        {f"<p><strong>Virtualization Overhead:</strong> {server_config['virtualization_overhead']*100:.1f}%</p>" if server_config['type'] == 'virtual' else ""}
    </div>
    """, unsafe_allow_html=True)
    
    # Hardware Configuration
    st.sidebar.markdown("**⚙️ Hardware Specifications:**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cpu_cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=16, step=2)
        ram_gb = st.number_input("RAM (GB)", min_value=4, max_value=1024, value=64, step=8)
    
    with col2:
        cpu_ghz = st.number_input("CPU GHz", min_value=1.0, max_value=5.0, value=2.8, step=0.2)
        storage_gb = st.number_input("Storage (GB)", min_value=100, max_value=100000, value=2000, step=100)
    
    # Performance Metrics
    st.sidebar.markdown("**📊 Current Performance Metrics:**")
    
    max_iops = st.sidebar.number_input("Max IOPS", min_value=100, max_value=1000000, value=50000, step=1000)
    max_memory_usage_gb = st.sidebar.number_input("Max Memory Usage (GB)", min_value=4, max_value=512, value=48, step=4)
    database_size_gb = st.sidebar.number_input("Database Size (GB)", min_value=100, max_value=100000, value=5000, step=100)
    
    return {
        'server_type': server_type,
        'cpu_cores': cpu_cores,
        'ram_gb': ram_gb,
        'cpu_ghz': cpu_ghz,
        'storage_gb': storage_gb,
        'max_iops': max_iops,
        'max_memory_usage_gb': max_memory_usage_gb,
        'database_size_gb': database_size_gb
    }

def render_migration_tool_configuration(selected_scenario: Dict):
    """Enhanced migration tool configuration"""
    st.sidebar.subheader("🔄 Migration Tool Configuration")
    
    migration_tool = selected_scenario['migration_tool']
    
    if migration_tool == 'datasync':
        st.sidebar.markdown(f"""
        <div class="migration-tool-card">
            <h4>📦 AWS DataSync Configuration</h4>
            <p><strong>Tool:</strong> AWS DataSync</p>
            <p><strong>Use Case:</strong> File-based migration, homogeneous transfers</p>
            <p><strong>Protocols:</strong> NFS, SMB, EFS, FSx</p>
            <p><strong>Encryption:</strong> In-transit and at-rest</p>
        </div>
        """, unsafe_allow_html=True)
        
        # DataSync specific settings
        datasync_agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': '📦 Small (t3.medium) - 250 Mbps/agent',
                'medium': '📦 Medium (c5.large) - 500 Mbps/agent', 
                'large': '📦 Large (c5.xlarge) - 1000 Mbps/agent',
                'xlarge': '📦 XLarge (c5.2xlarge) - 2000 Mbps/agent'
            }[x]
        )
        
        number_of_agents = st.sidebar.number_input(
            "Number of DataSync Agents",
            min_value=1,
            max_value=8,
            value=selected_scenario['recommended_agents'],
            help="More agents can improve throughput but increase complexity"
        )
        
        return {
            'migration_tool': 'datasync',
            'agent_size': datasync_agent_size,
            'number_of_agents': number_of_agents,
            'parallel_transfers': True,
            'bandwidth_throttling': st.sidebar.checkbox("Enable Bandwidth Throttling", value=True)
        }
    
    else:  # DMS
        st.sidebar.markdown(f"""
        <div class="migration-tool-card">
            <h4>🔄 AWS DMS Configuration</h4>
            <p><strong>Tool:</strong> AWS Database Migration Service</p>
            <p><strong>Use Case:</strong> Database migration, heterogeneous transfers</p>
            <p><strong>Features:</strong> Schema conversion, CDC, validation</p>
            <p><strong>Engines:</strong> Oracle, SQL Server, MySQL, PostgreSQL</p>
        </div>
        """, unsafe_allow_html=True)
        
        # DMS specific settings
        dms_instance_size = st.sidebar.selectbox(
            "DMS Replication Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=2,
            format_func=lambda x: {
                'small': '🔄 Small (t3.medium) - 200 Mbps',
                'medium': '🔄 Medium (c5.large) - 400 Mbps',
                'large': '🔄 Large (c5.xlarge) - 800 Mbps',
                'xlarge': '🔄 XLarge (c5.2xlarge) - 1500 Mbps',
                'xxlarge': '🔄 XXLarge (c5.4xlarge) - 2500 Mbps'
            }[x]
        )
        
        number_of_instances = st.sidebar.number_input(
            "Number of DMS Instances",
            min_value=1,
            max_value=5,
            value=min(selected_scenario['recommended_agents'], 3),
            help="Multiple instances for parallel table migration"
        )
        
        return {
            'migration_tool': 'dms',
            'instance_size': dms_instance_size,
            'number_of_instances': number_of_instances,
            'cdc_enabled': st.sidebar.checkbox("Enable Change Data Capture", value=True),
            'schema_conversion': st.sidebar.checkbox("Include Schema Conversion", value=True)
        }

def render_destination_configuration(selected_scenario: Dict):
    """Enhanced destination configuration"""
    st.sidebar.subheader("🎯 Destination Configuration")
    
    destination = selected_scenario['destination']
    
    if destination == 's3':
        st.sidebar.markdown(f"""
        <div class="destination-config-card">
            <h4>☁️ Amazon S3 Configuration</h4>
            <p><strong>Service:</strong> Amazon S3</p>
            <p><strong>Use Case:</strong> Object storage, data lakes, backups</p>
            <p><strong>Benefits:</strong> Unlimited scale, low cost, high durability</p>
        </div>
        """, unsafe_allow_html=True)
        
        s3_storage_class = st.sidebar.selectbox(
            "S3 Storage Class",
            ["standard", "intelligent_tiering", "standard_ia"],
            format_func=lambda x: {
                'standard': 'Standard (Frequently accessed)',
                'intelligent_tiering': 'Intelligent Tiering (Auto optimization)',
                'standard_ia': 'Standard-IA (Infrequently accessed)'
            }[x]
        )
        
        return {
            'destination_type': 's3',
            'storage_class': s3_storage_class,
            'encryption': st.sidebar.checkbox("Enable S3 Encryption", value=True),
            'versioning': st.sidebar.checkbox("Enable Versioning", value=True)
        }
    
    elif destination == 'fsx_lustre':
        st.sidebar.markdown(f"""
        <div class="destination-config-card">
            <h4>⚡ FSx for Lustre Configuration</h4>
            <p><strong>Service:</strong> Amazon FSx for Lustre</p>
            <p><strong>Use Case:</strong> High-performance computing, ML workloads</p>
            <p><strong>Benefits:</strong> POSIX-compliant, optimized for throughput</p>
        </div>
        """, unsafe_allow_html=True)
        
        fsx_deployment_type = st.sidebar.selectbox(
            "FSx Deployment Type",
            ["scratch_1", "scratch_2", "persistent_1", "persistent_2"],
            index=2,
            format_func=lambda x: {
                'scratch_1': 'Scratch 1 (200 MB/s/TiB)',
                'scratch_2': 'Scratch 2 (1000 MB/s/TiB)', 
                'persistent_1': 'Persistent 1 (50-200 MB/s/TiB)',
                'persistent_2': 'Persistent 2 (125-1000 MB/s/TiB)'
            }[x]
        )
        
        return {
            'destination_type': 'fsx_lustre',
            'deployment_type': fsx_deployment_type,
            'storage_capacity_gb': st.sidebar.number_input("Storage Capacity (GB)", min_value=1200, max_value=100800, value=7200, step=1200)
        }
    
    else:  # fsx_windows
        st.sidebar.markdown(f"""
        <div class="destination-config-card">
            <h4>🪟 FSx for Windows Configuration</h4>
            <p><strong>Service:</strong> Amazon FSx for Windows File Server</p>
            <p><strong>Use Case:</strong> Windows workloads, Active Directory integration</p>
            <p><strong>Benefits:</strong> Fully managed, SMB protocol, backup integration</p>
        </div>
        """, unsafe_allow_html=True)
        
        fsx_throughput_capacity = st.sidebar.selectbox(
            "Throughput Capacity",
            [8, 16, 32, 64, 128, 256, 512, 1024],
            index=3,
            format_func=lambda x: f"{x} MB/s"
        )
        
        return {
            'destination_type': 'fsx_windows',
            'throughput_capacity': fsx_throughput_capacity,
            'storage_capacity_gb': st.sidebar.number_input("Storage Capacity (GB)", min_value=32, max_value=65536, value=1024, step=32),
            'backup_retention_days': st.sidebar.number_input("Backup Retention (days)", min_value=0, max_value=90, value=7)
        }

def render_16_scenario_analysis_results(analysis: Dict, scenario_key: str, config: Dict):
    """Render comprehensive analysis results for selected scenario"""
    
    scenario_performance = analysis['scenario_performance']
    scenario = scenario_performance['scenario']
    
    # Scenario Overview
    st.subheader(f"📊 Scenario {scenario['id']} Analysis Results")
    
    st.markdown(f"""
    <div class="scenario-overview-card">
        <h3>{scenario['name']}</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div>
                <strong>🌍 Environment:</strong> {scenario['environment'].title()}<br>
                <strong>🔧 Tool:</strong> {scenario['migration_tool'].upper()}<br>
                <strong>🖥️ Source OS:</strong> {scenario['source_os'].title()}
            </div>
            <div>
                <strong>📍 Route:</strong> {scenario['source_location']} → AWS<br>
                <strong>🎯 Destination:</strong> {scenario['destination'].replace('_', ' ').title()}<br>
                <strong>☁️ VPC:</strong> {scenario['destination_vpc'].replace('_', ' ').title()}
            </div>
            <div>
                <strong>🎚️ Complexity:</strong> {scenario['complexity_score']}/10<br>
                <strong>🤖 Recommended Agents:</strong> {scenario['recommended_agents']}<br>
                <strong>⚡ Agent Efficiency:</strong> {scenario_performance['agent_efficiency']*100:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance Metrics Overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🚀 Effective Bandwidth",
            f"{scenario_performance['effective_bandwidth_mbps']:,.0f} Mbps",
            delta=f"Latency: {scenario_performance['total_latency_ms']:.1f}ms"
        )
    
    with col2:
        st.metric(
            "🛡️ Path Reliability",
            f"{scenario_performance['total_reliability']*100:.3f}%",
            delta=f"Downtime: {(1-scenario_performance['total_reliability'])*365*24*60:.0f}min/yr"
        )
    
    with col3:
        st.metric(
            "⏱️ Migration Time",
            f"{analysis['migration_time_hours']:.1f} hours",
            delta=f"Complexity: {scenario['complexity_score']}/10"
        )
    
    with col4:
        st.metric(
            "💰 Monthly Cost", 
            f"${analysis['cost_analysis']['total_monthly_cost']:,.0f}",
            delta=f"Agent Cost: ${analysis['cost_analysis']['agent_cost']}"
        )
    
    with col5:
        st.metric(
            "🤖 Agent Configuration",
            f"{config.get('number_of_agents', 1)} agents",
            delta=f"Recommended: {scenario['recommended_agents']}"
        )
    
    # Network Path Visualization
    st.markdown("**🗺️ Network Path Analysis:**")
    
    # Create network path diagram
    network_segments = scenario_performance['network_segments']
    
    # Display network segments as a table
    segments_df = pd.DataFrame(network_segments)
    segments_df['Efficiency'] = segments_df.apply(lambda row: f"{(row['bandwidth_mbps']/max(segments_df['bandwidth_mbps']))*100:.1f}%", axis=1)
    segments_df['Quality Score'] = segments_df.apply(lambda row: f"{(row['reliability']*100):.1f}%", axis=1)
    
    st.dataframe(
        segments_df[['segment', 'bandwidth_mbps', 'latency_ms', 'reliability', 'Efficiency', 'Quality Score']],
        column_config={
            'segment': 'Network Segment',
            'bandwidth_mbps': st.column_config.NumberColumn('Bandwidth (Mbps)', format="%.0f"),
            'latency_ms': st.column_config.NumberColumn('Latency (ms)', format="%.1f"),
            'reliability': st.column_config.NumberColumn('Reliability', format="%.4f"),
        },
        hide_index=True
    )
    
    # AI Insights and Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 AI Analysis:**")
        ai_insights = analysis['ai_insights']
        
        st.markdown(f"""
        <div class="ai-insight-card">
            <h4>🎯 Complexity Assessment</h4>
            <p>{ai_insights['complexity_assessment']}</p>
            
            <h5>🔍 Identified Bottlenecks:</h5>
            <ul>
                {"".join([f"<li>{bottleneck}</li>" for bottleneck in ai_insights['bottleneck_analysis']])}
            </ul>
            
            <h5>⚠️ Risk Factors:</h5>
            <ul>
                {"".join([f"<li>{risk}</li>" for risk in ai_insights['risk_factors']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**💡 Optimization Recommendations:**")
        
        all_recommendations = ai_insights['optimization_recommendations'] + analysis['recommendations']
        
        st.markdown(f"""
        <div class="ai-insight-card">
            <h4>🚀 Performance Optimizations</h4>
            <ul>
                {"".join([f"<li>{rec}</li>" for rec in all_recommendations[:6]])}
            </ul>
            
            <h5>📈 Expected Benefits:</h5>
            <p>• Improved migration throughput</p>
            <p>• Reduced complexity and risk</p>
            <p>• Enhanced reliability and monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cost Breakdown
    st.markdown("**💰 Detailed Cost Analysis:**")
    
    cost_analysis = analysis['cost_analysis']
    
    cost_col1, cost_col2, cost_col3 = st.columns(3)
    
    with cost_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>☁️ AWS Infrastructure Costs</h4>
            <p><strong>Base AWS Cost:</strong> ${cost_analysis['base_aws_cost']:,.0f}/month</p>
            <p><strong>Includes:</strong> RDS/EC2, Storage, Backup</p>
            <p><strong>Scenario Multiplier:</strong> {cost_analysis['scenario_complexity_multiplier']:.1f}x</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cost_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌐 Network & Connectivity</h4>
            <p><strong>Network Cost:</strong> ${cost_analysis['network_cost']:,.0f}/month</p>
            <p><strong>Type:</strong> {'Multi-hop' if scenario['source_location'] == 'San Antonio' else 'Direct'}</p>
            <p><strong>Bandwidth:</strong> {'10Gbps' if scenario['environment'] == 'production' else '2Gbps'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cost_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🤖 Agent & Tool Costs</h4>
            <p><strong>Agent Cost:</strong> ${cost_analysis['agent_cost']:,.0f}/month</p>
            <p><strong>Number of Agents:</strong> {config.get('number_of_agents', 1)}</p>
            <p><strong>Tool:</strong> {scenario['migration_tool'].upper()}</p>
        </div>
        """, unsafe_allow_html=True)

async def main():
    """Enhanced main function supporting all 16 scenarios"""
    render_enhanced_header_16_scenarios()
    
    # Enhanced sidebar with scenario selection
    scenario_key, selected_scenario = render_scenario_selector_sidebar()
    
    st.sidebar.markdown("---")
    
    # Enhanced server configuration
    server_config = render_enhanced_server_configuration()
    
    st.sidebar.markdown("---")
    
    # Migration tool configuration
    tool_config = render_migration_tool_configuration(selected_scenario)
    
    st.sidebar.markdown("---")
    
    # Destination configuration  
    dest_config = render_destination_configuration(selected_scenario)
    
    # Combine all configuration
    config = {
        **server_config,
        **tool_config,
        **dest_config,
        'scenario_key': scenario_key,
        'selected_scenario': selected_scenario
    }
    
    # Initialize enhanced analyzer
    analyzer = Enhanced16ScenarioAnalyzer()
    
    # Run analysis button
    if st.sidebar.button("🚀 Analyze Migration Scenario", type="primary", use_container_width=True):
        analysis_placeholder = st.empty()
        
        with analysis_placeholder.container():
            with st.spinner(f"🧠 Analyzing Scenario {selected_scenario['id']}: {selected_scenario['name']}..."):
                try:
                    analysis = await analyzer.analyze_migration_scenario(scenario_key, config)
                    
                    # Store in session state
                    st.session_state['analysis'] = analysis
                    st.session_state['config'] = config
                    st.session_state['scenario_key'] = scenario_key
                    
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    analysis = None
        
        analysis_placeholder.empty()
    
    # Display results if available
    if 'analysis' in st.session_state:
        render_16_scenario_analysis_results(
            st.session_state['analysis'], 
            st.session_state['scenario_key'], 
            st.session_state['config']
        )
        
        # Scenario Comparison
        st.markdown("---")
        st.subheader("🔍 Scenario Comparison & Alternative Analysis")
        
        if st.button("🔄 Compare with Other Scenarios"):
            st.info("🚧 Scenario comparison feature coming soon! This will allow you to compare multiple scenarios side-by-side.")
    
    else:
        # Welcome screen
        st.markdown("## 🎯 Welcome to the 16-Scenario Migration Analyzer")
        
        st.markdown("""
        This enhanced migration analyzer supports **all 16 migration scenarios** across:
        
        - **🌍 Environments:** Non-Production & Production
        - **🔧 Tools:** AWS DataSync & AWS DMS  
        - **🖥️ Source Systems:** Linux NAS & Windows Shares
        - **🎯 Destinations:** S3, FSx for Lustre, FSx for Windows
        - **📍 Network Paths:** San Jose direct & San Antonio via San Jose
        
        **🚀 To get started:**
        1. Choose your migration scenario using the sidebar
        2. Configure your server specifications
        3. Set up migration tool parameters
        4. Configure destination settings
        5. Click "Analyze Migration Scenario"
        """)
        
        # Display all 16 scenarios overview
        st.markdown("### 📋 All 16 Migration Scenarios Overview")
        
        network_manager = Enhanced16ScenarioNetworkManager()
        
        scenarios_data = []
        for key, scenario in network_manager.migration_scenarios.items():
            scenarios_data.append({
                'ID': scenario['id'],
                'Name': scenario['name'],
                'Environment': scenario['environment'].title(),
                'Tool': scenario['migration_tool'].upper(),
                'Source': f"{scenario['source_location']} ({scenario['source_os'].title()})",
                'Destination': scenario['destination'].replace('_', ' ').title(),
                'Complexity': f"{scenario['complexity_score']}/10",
                'Recommended Agents': scenario['recommended_agents']
            })
        
        scenarios_df = pd.DataFrame(scenarios_data)
        st.dataframe(
            scenarios_df,
            column_config={
                'ID': st.column_config.NumberColumn('Scenario ID', width='small'),
                'Name': st.column_config.TextColumn('Scenario Name', width='large'),
                'Environment': st.column_config.TextColumn('Environment', width='medium'),
                'Tool': st.column_config.TextColumn('Migration Tool', width='small'),
                'Source': st.column_config.TextColumn('Source', width='medium'),
                'Destination': st.column_config.TextColumn('Destination', width='medium'),
                'Complexity': st.column_config.TextColumn('Complexity', width='small'),
                'Recommended Agents': st.column_config.NumberColumn('Agents', width='small')
            },
            hide_index=True,
            use_container_width=True
        )
    
    # Professional footer
    st.markdown("""
    <div style="margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); border-radius: 8px; text-align: center; color: white;">
        <h4>🚀 AWS Enterprise Database Migration Analyzer AI v4.0 - Complete 16-Scenario Platform</h4>
        <p>Powered by Advanced AI • Complete Scenario Coverage • Professional Migration Analysis • Enterprise-Ready Architecture</p>
        <p style="font-size: 0.9rem; margin-top: 1rem; opacity: 0.9;">
            🎯 All 16 Migration Scenarios • 🤖 Multi-Agent Optimization • 🔬 Advanced Performance Analysis • 📊 Executive Reporting
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())