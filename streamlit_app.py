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

# Page configuration
st.set_page_config(
    page_title="AWS Enterprise Database Migration Analyzer v2.0",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with AWS theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF9900 0%, #232F3E 50%, #4ECDC4 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF9900;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    .os-comparison-card {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(255,152,0,0.1);
    }
    
    .aws-migration-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(76,175,80,0.1);
    }
    
    .network-path-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(33,150,243,0.1);
    }
    
    .aws-recommendation-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #9c27b0;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(156,39,176,0.1);
    }
    
    .agent-sizing-card {
        background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #009688;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(0,150,136,0.1);
    }
    
    .pricing-card {
        background: linear-gradient(135deg, #fff9c4 0%, #f9fbe7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #689f38;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(104,159,56,0.1);
    }
    
    .performance-card {
        background: linear-gradient(135deg, #fce4ec 0%, #f8bbd9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #e91e63;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(233,30,99,0.1);
    }
    
    .onprem-performance-card {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #3f51b5;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(63,81,181,0.1);
    }
    
    .aws-performance-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(255,152,0,0.1);
    }
    
    .performance-delta {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        margin-left: 8px;
    }
    
    .delta-positive { background-color: #d4edda; color: #155724; }
    .delta-negative { background-color: #f8d7da; color: #721c24; }
    .delta-neutral { background-color: #d1ecf1; color: #0c5460; }
    
    .rds-recommendation {
        background: linear-gradient(135deg, #ff9900 0%, #ff6600 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .ec2-recommendation {
        background: linear-gradient(135deg, #232f3e 0%, #131a22 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class OSPerformanceManager:
    """Manage Operating System performance characteristics"""
    
    def __init__(self):
        self.operating_systems = {
            'windows_server_2019': {
                'name': 'Windows Server 2019',
                'cpu_efficiency': 0.92,
                'memory_efficiency': 0.88,
                'io_efficiency': 0.85,
                'network_efficiency': 0.90,
                'virtualization_overhead': 0.12,
                'database_optimizations': {
                    'mysql': 0.88,
                    'postgresql': 0.85,
                    'oracle': 0.95,
                    'sqlserver': 0.98,
                    'mongodb': 0.87
                },
                'licensing_cost_factor': 2.5,
                'management_complexity': 0.6,
                'security_overhead': 0.08
            },
            'windows_server_2022': {
                'name': 'Windows Server 2022',
                'cpu_efficiency': 0.95,
                'memory_efficiency': 0.92,
                'io_efficiency': 0.90,
                'network_efficiency': 0.93,
                'virtualization_overhead': 0.10,
                'database_optimizations': {
                    'mysql': 0.90,
                    'postgresql': 0.88,
                    'oracle': 0.97,
                    'sqlserver': 0.99,
                    'mongodb': 0.89
                },
                'licensing_cost_factor': 3.0,
                'management_complexity': 0.5,
                'security_overhead': 0.06
            },
            'rhel_8': {
                'name': 'Red Hat Enterprise Linux 8',
                'cpu_efficiency': 0.96,
                'memory_efficiency': 0.94,
                'io_efficiency': 0.95,
                'network_efficiency': 0.95,
                'virtualization_overhead': 0.06,
                'database_optimizations': {
                    'mysql': 0.95,
                    'postgresql': 0.97,
                    'oracle': 0.93,
                    'sqlserver': 0.85,
                    'mongodb': 0.96
                },
                'licensing_cost_factor': 1.5,
                'management_complexity': 0.7,
                'security_overhead': 0.04
            },
            'rhel_9': {
                'name': 'Red Hat Enterprise Linux 9',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.96,
                'io_efficiency': 0.97,
                'network_efficiency': 0.97,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.97,
                    'postgresql': 0.98,
                    'oracle': 0.95,
                    'sqlserver': 0.87,
                    'mongodb': 0.97
                },
                'licensing_cost_factor': 1.8,
                'management_complexity': 0.6,
                'security_overhead': 0.03
            },
            'ubuntu_20_04': {
                'name': 'Ubuntu Server 20.04 LTS',
                'cpu_efficiency': 0.97,
                'memory_efficiency': 0.95,
                'io_efficiency': 0.96,
                'network_efficiency': 0.96,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.96,
                    'postgresql': 0.98,
                    'oracle': 0.90,
                    'sqlserver': 0.82,
                    'mongodb': 0.97
                },
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.8,
                'security_overhead': 0.03
            },
            'ubuntu_22_04': {
                'name': 'Ubuntu Server 22.04 LTS',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.97,
                'io_efficiency': 0.98,
                'network_efficiency': 0.98,
                'virtualization_overhead': 0.04,
                'database_optimizations': {
                    'mysql': 0.98,
                    'postgresql': 0.99,
                    'oracle': 0.92,
                    'sqlserver': 0.84,
                    'mongodb': 0.98
                },
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.7,
                'security_overhead': 0.02
            }
        }
    
    def calculate_os_performance_impact(self, os_type: str, platform_type: str, database_engine: str) -> Dict:
        """Calculate OS performance impact on database migration"""
        
        os_config = self.operating_systems[os_type]
        
        # Base OS efficiency
        base_efficiency = (
            os_config['cpu_efficiency'] * 0.3 +
            os_config['memory_efficiency'] * 0.25 +
            os_config['io_efficiency'] * 0.25 +
            os_config['network_efficiency'] * 0.2
        )
        
        # Database-specific optimization
        db_optimization = os_config['database_optimizations'][database_engine]
        
        # Virtualization impact
        if platform_type == 'vmware':
            virtualization_penalty = os_config['virtualization_overhead']
            total_efficiency = base_efficiency * db_optimization * (1 - virtualization_penalty)
        else:
            total_efficiency = base_efficiency * db_optimization
        
        # Platform-specific adjustments
        if platform_type == 'physical':
            # Physical servers get slight boost for Windows due to direct hardware access
            if 'windows' in os_type:
                total_efficiency *= 1.02
            else:
                total_efficiency *= 1.05  # Linux generally better on physical
        
        return {
            'total_efficiency': total_efficiency,
            'base_efficiency': base_efficiency,
            'db_optimization': db_optimization,
            'virtualization_overhead': os_config['virtualization_overhead'] if platform_type == 'vmware' else 0,
            'licensing_cost_factor': os_config['licensing_cost_factor'],
            'management_complexity': os_config['management_complexity'],
            'security_overhead': os_config['security_overhead'],
            'cpu_efficiency': os_config['cpu_efficiency'],
            'memory_efficiency': os_config['memory_efficiency'],
            'io_efficiency': os_config['io_efficiency'],
            'network_efficiency': os_config['network_efficiency']
        }

class AgentSizingManager:
    """Manage AWS DataSync and DMS agent sizing"""
    
    def __init__(self):
        self.datasync_agents = {
            'small': {
                'name': 'Small Agent (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps': 250,
                'max_concurrent_tasks': 10,
                'cost_per_hour': 0.0416,
                'recommended_for': 'Up to 1TB databases, <100 Mbps network'
            },
            'medium': {
                'name': 'Medium Agent (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps': 500,
                'max_concurrent_tasks': 25,
                'cost_per_hour': 0.085,
                'recommended_for': '1-5TB databases, 100-500 Mbps network'
            },
            'large': {
                'name': 'Large Agent (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps': 1000,
                'max_concurrent_tasks': 50,
                'cost_per_hour': 0.17,
                'recommended_for': '5-20TB databases, 500Mbps-1Gbps network'
            },
            'xlarge': {
                'name': 'XLarge Agent (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps': 2000,
                'max_concurrent_tasks': 100,
                'cost_per_hour': 0.34,
                'recommended_for': '>20TB databases, >1Gbps network'
            }
        }
        
        self.dms_agents = {
            'small': {
                'name': 'Small DMS Instance (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps': 200,
                'max_concurrent_tasks': 5,
                'cost_per_hour': 0.0416,
                'recommended_for': 'Up to 500GB databases, simple schemas'
            },
            'medium': {
                'name': 'Medium DMS Instance (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps': 400,
                'max_concurrent_tasks': 10,
                'cost_per_hour': 0.085,
                'recommended_for': '500GB-2TB databases, moderate complexity'
            },
            'large': {
                'name': 'Large DMS Instance (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps': 800,
                'max_concurrent_tasks': 20,
                'cost_per_hour': 0.17,
                'recommended_for': '2-10TB databases, complex schemas'
            },
            'xlarge': {
                'name': 'XLarge DMS Instance (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps': 1500,
                'max_concurrent_tasks': 40,
                'cost_per_hour': 0.34,
                'recommended_for': '>10TB databases, very complex schemas'
            },
            'xxlarge': {
                'name': 'XXLarge DMS Instance (c5.4xlarge)',
                'vcpu': 16,
                'memory_gb': 32,
                'max_throughput_mbps': 2500,
                'max_concurrent_tasks': 80,
                'cost_per_hour': 0.68,
                'recommended_for': '>50TB databases, enterprise workloads'
            }
        }
    
    def recommend_agent_size(self, tool_type: str, database_size_gb: int, 
                           network_bandwidth_mbps: int, migration_complexity: str) -> str:
        """Recommend appropriate agent size based on requirements"""
        
        agents = self.datasync_agents if tool_type == 'datasync' else self.dms_agents
        
        # Size selection logic
        if database_size_gb < 1000 and network_bandwidth_mbps < 250:
            return 'small'
        elif database_size_gb < 5000 and network_bandwidth_mbps < 750:
            if migration_complexity == 'high' and tool_type == 'dms':
                return 'large'
            return 'medium'
        elif database_size_gb < 20000 and network_bandwidth_mbps < 1500:
            return 'large'
        elif database_size_gb < 50000:
            return 'xlarge'
        else:
            return 'xxlarge' if tool_type == 'dms' else 'xlarge'

class EnhancedNetworkPathManager:
    """Enhanced network path manager with realistic enterprise scenarios"""
    
    def __init__(self):
        self.network_paths = {
            'nonprod_sj_linux_nas': {
                'name': 'Non-Prod: San Jose Linux NAS + Jump Server → AWS S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,  # 1 Gbps internal
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0
                    },
                    {
                        'name': 'Linux Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,  # 2 Gbps DX
                        'latency_ms': 15,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0
                    }
                ]
            },
            'nonprod_sj_windows_share': {
                'name': 'Non-Prod: San Jose Windows Share + Jump Server → AWS S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 1000,  # 1 Gbps internal
                        'latency_ms': 3,  # Slightly higher due to Windows overhead
                        'reliability': 0.997,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0
                    },
                    {
                        'name': 'Windows Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,  # 2 Gbps DX
                        'latency_ms': 18,  # Slightly higher due to Windows network stack
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0
                    }
                ]
            },
            'prod_sa_linux_nas': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS Production VPC S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,  # 1 Gbps internal
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,  # 10 Gbps
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,  # 10 Gbps DX
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0
                    }
                ]
            },
            'prod_sa_windows_share': {
                'name': 'Prod: San Antonio Windows Share → San Jose → AWS Production VPC S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'San Antonio Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 1000,  # 1 Gbps internal
                        'latency_ms': 2,
                        'reliability': 0.998,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,  # 10 Gbps shared
                        'latency_ms': 15,  # Higher due to Windows network processing
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,  # 10 Gbps DX
                        'latency_ms': 10,  # Slightly higher for Windows
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0
                    }
                ]
            }
        }
    
    def calculate_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """Calculate network path performance with realistic congestion modeling"""
        
        path = self.network_paths[path_key]
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        # Calculate end-to-end performance
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        
        adjusted_segments = []
        
        for segment in path['segments']:
            # Base metrics
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Time-of-day and congestion adjustments
            if segment['connection_type'] == 'internal_lan':
                # Internal LAN - minimal variation
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.1
                else:
                    congestion_factor = 0.95
            elif segment['connection_type'] == 'private_line':
                # Private lines - moderate business hours impact
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.2
                else:
                    congestion_factor = 0.9
            elif segment['connection_type'] == 'direct_connect':
                # DX connections - very stable, minimal variation
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.05
                else:
                    congestion_factor = 0.98
            else:
                congestion_factor = 1.0
            
            # Apply congestion
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # OS-specific adjustments
            if path['os_type'] == 'windows' and segment['connection_type'] != 'internal_lan':
                # Windows has slightly higher network overhead
                effective_bandwidth *= 0.95
                effective_latency *= 1.1
            
            # Accumulate metrics
            total_latency += effective_latency
            min_bandwidth = min(min_bandwidth, effective_bandwidth)
            total_reliability *= segment_reliability
            total_cost_factor += segment['cost_factor']
            
            adjusted_segments.append({
                **segment,
                'effective_bandwidth_mbps': effective_bandwidth,
                'effective_latency_ms': effective_latency
            })
        
        # Calculate quality scores
        latency_score = max(0, 100 - (total_latency * 2))  # Penalize high latency
        bandwidth_score = min(100, (min_bandwidth / 1000) * 20)  # Score based on Gbps
        reliability_score = total_reliability * 100
        
        # Overall network quality with weighted factors
        network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        
        return {
            'path_name': path['name'],
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': network_quality,
            'total_cost_factor': total_cost_factor,
            'segments': adjusted_segments,
            'environment': path['environment'],
            'os_type': path['os_type'],
            'storage_type': path['storage_type']
        }

class EnhancedAWSMigrationManager:
    """Enhanced AWS migration manager with advanced sizing and pricing"""
    
    def __init__(self):
        self.migration_types = {
            'homogeneous': {
                'name': 'Homogeneous Migration',
                'description': 'Same database engine (uses DataSync for file-based, native replication for binary)',
                'complexity_factor': 0.3,
                'time_factor': 0.7,
                'risk_factor': 0.2,
                'preferred_tools': ['aws_datasync', 'native_replication'],
                'schema_conversion_required': False,
                'application_changes_required': False
            },
            'heterogeneous': {
                'name': 'Heterogeneous Migration',
                'description': 'Different database engines (requires DMS and SCT)',
                'complexity_factor': 0.8,
                'time_factor': 1.4,
                'risk_factor': 0.7,
                'preferred_tools': ['aws_dms', 'aws_sct'],
                'schema_conversion_required': True,
                'application_changes_required': True
            }
        }
        
        # Enhanced AWS pricing data (simulated real-time pricing)
        self.aws_pricing = {
            'us-west-2': {
                'ec2_instances': {
                    't3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416},
                    't3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.0832},
                    't3.xlarge': {'vcpu': 4, 'memory': 16, 'cost_per_hour': 0.1664},
                    'c5.large': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085},
                    'c5.xlarge': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17},
                    'c5.2xlarge': {'vcpu': 8, 'memory': 16, 'cost_per_hour': 0.34},
                    'c5.4xlarge': {'vcpu': 16, 'memory': 32, 'cost_per_hour': 0.68},
                    'r6i.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.252},
                    'r6i.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.504},
                    'r6i.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.008},
                    'r6i.4xlarge': {'vcpu': 16, 'memory': 128, 'cost_per_hour': 2.016},
                    'r6i.8xlarge': {'vcpu': 32, 'memory': 256, 'cost_per_hour': 4.032}
                },
                'rds_instances': {
                    'db.t3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.068},
                    'db.t3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.136},
                    'db.r6g.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.48},
                    'db.r6g.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.96},
                    'db.r6g.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.92},
                    'db.r6g.4xlarge': {'vcpu': 16, 'memory': 128, 'cost_per_hour': 3.84},
                    'db.r6g.8xlarge': {'vcpu': 32, 'memory': 256, 'cost_per_hour': 7.68}
                },
                'storage': {
                    'gp3': {'cost_per_gb_month': 0.08, 'iops_included': 3000},
                    'io1': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065},
                    'io2': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065}
                }
            }
        }
    
    def recommend_aws_sizing(self, on_prem_config: Dict) -> Dict:
        """AI-enhanced AWS sizing recommendations based on on-premises configuration"""
        
        # Extract on-premises specs
        cpu_cores = on_prem_config['cpu_cores']
        ram_gb = on_prem_config['ram_gb']
        database_size_gb = on_prem_config['database_size_gb']
        performance_req = on_prem_config.get('performance_requirements', 'standard')
        database_engine = on_prem_config['database_engine']
        environment = on_prem_config.get('environment', 'non-production')
        
        # AI-based sizing logic
        recommendations = {}
        
        # RDS Recommendations
        rds_recommendations = self._calculate_rds_sizing(
            cpu_cores, ram_gb, database_size_gb, performance_req, database_engine, environment
        )
        
        # EC2 Recommendations
        ec2_recommendations = self._calculate_ec2_sizing(
            cpu_cores, ram_gb, database_size_gb, performance_req, database_engine, environment
        )
        
        # Reader/Writer Configuration
        reader_writer_config = self._calculate_reader_writer_config(
            database_size_gb, performance_req, environment
        )
        
        return {
            'rds_recommendations': rds_recommendations,
            'ec2_recommendations': ec2_recommendations,
            'reader_writer_config': reader_writer_config,
            'deployment_recommendation': self._recommend_deployment_type(
                database_size_gb, performance_req, database_engine, environment
            )
        }
    
    def _calculate_rds_sizing(self, cpu_cores: int, ram_gb: int, database_size_gb: int,
                            performance_req: str, database_engine: str, environment: str) -> Dict:
        """Calculate RDS instance sizing with AI-based recommendations"""
        
        # Base sizing multipliers for cloud migration
        cpu_multiplier = 1.2 if performance_req == 'high' else 1.0
        memory_multiplier = 1.3 if database_engine in ['oracle', 'postgresql'] else 1.1
        
        # Required cloud resources
        required_vcpu = max(2, int(cpu_cores * cpu_multiplier))
        required_memory = max(8, int(ram_gb * memory_multiplier))
        
        # Instance selection logic
        rds_instances = self.aws_pricing['us-west-2']['rds_instances']
        
        # Find best fit instance
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in rds_instances.items():
            if specs['vcpu'] >= required_vcpu and specs['memory'] >= required_memory:
                # Calculate efficiency score (lower is better)
                cpu_waste = specs['vcpu'] - required_vcpu
                memory_waste = specs['memory'] - required_memory
                cost_factor = specs['cost_per_hour']
                
                score = (cpu_waste * 0.3 + memory_waste * 0.001 + cost_factor * 0.5)
                
                if score < best_score:
                    best_score = score
                    best_instance = instance_type
        
        if not best_instance:
            best_instance = 'db.r6g.8xlarge'  # Fallback for very large requirements
        
        # Storage recommendations
        storage_size_gb = max(database_size_gb * 1.5, 100)  # 50% overhead
        storage_type = 'io1' if database_size_gb > 5000 or performance_req == 'high' else 'gp3'
        
        # Calculate monthly costs
        instance_cost = rds_instances[best_instance]['cost_per_hour'] * 24 * 30
        storage_cost = storage_size_gb * self.aws_pricing['us-west-2']['storage'][storage_type]['cost_per_gb_month']
        
        return {
            'primary_instance': best_instance,
            'instance_specs': rds_instances[best_instance],
            'storage_type': storage_type,
            'storage_size_gb': storage_size_gb,
            'monthly_instance_cost': instance_cost,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': instance_cost + storage_cost,
            'multi_az': environment == 'production',
            'backup_retention_days': 30 if environment == 'production' else 7
        }
    
    def _calculate_ec2_sizing(self, cpu_cores: int, ram_gb: int, database_size_gb: int,
                            performance_req: str, database_engine: str, environment: str) -> Dict:
        """Calculate EC2 instance sizing for self-managed databases"""
        
        # EC2 needs more overhead for OS and database management
        cpu_multiplier = 1.4 if performance_req == 'high' else 1.2
        memory_multiplier = 1.5 if database_engine in ['oracle', 'postgresql'] else 1.3
        
        required_vcpu = max(2, int(cpu_cores * cpu_multiplier))
        required_memory = max(8, int(ram_gb * memory_multiplier))
        
        # Instance selection
        ec2_instances = self.aws_pricing['us-west-2']['ec2_instances']
        
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in ec2_instances.items():
            if specs['vcpu'] >= required_vcpu and specs['memory'] >= required_memory:
                cpu_waste = specs['vcpu'] - required_vcpu
                memory_waste = specs['memory'] - required_memory
                cost_factor = specs['cost_per_hour']
                
                score = (cpu_waste * 0.3 + memory_waste * 0.001 + cost_factor * 0.5)
                
                if score < best_score:
                    best_score = score
                    best_instance = instance_type
        
        if not best_instance:
            best_instance = 'r6i.8xlarge'
        
        # Storage sizing (more generous for EC2)
        storage_size_gb = max(database_size_gb * 2.0, 100)  # 100% overhead for OS, logs, backups
        storage_type = 'io2' if performance_req == 'high' else 'gp3'
        
        # Calculate costs
        instance_cost = ec2_instances[best_instance]['cost_per_hour'] * 24 * 30
        storage_cost = storage_size_gb * self.aws_pricing['us-west-2']['storage'][storage_type]['cost_per_gb_month']
        
        return {
            'primary_instance': best_instance,
            'instance_specs': ec2_instances[best_instance],
            'storage_type': storage_type,
            'storage_size_gb': storage_size_gb,
            'monthly_instance_cost': instance_cost,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': instance_cost + storage_cost,
            'ebs_optimized': True,
            'enhanced_networking': True
        }
    
    def _calculate_reader_writer_config(self, database_size_gb: int, performance_req: str, environment: str) -> Dict:
        """Calculate optimal reader/writer configuration"""
        
        # Start with single writer
        writers = 1
        readers = 0
        
        # Add readers based on size and performance requirements
        if database_size_gb > 1000:
            readers += 1
        if database_size_gb > 5000:
            readers += 1
        if database_size_gb > 20000:
            readers += 2
        
        # Performance-based scaling
        if performance_req == 'high':
            readers += 2
        
        # Environment-based scaling
        if environment == 'production':
            readers = max(readers, 2)  # Minimum 2 readers for production
            if database_size_gb > 50000:
                writers = 2  # Multi-writer for very large production DBs
        
        # Calculate read/write distribution
        total_capacity = writers + readers
        write_capacity_percent = (writers / total_capacity) * 100
        read_capacity_percent = (readers / total_capacity) * 100
        
        return {
            'writers': writers,
            'readers': readers,
            'total_instances': total_capacity,
            'write_capacity_percent': write_capacity_percent,
            'read_capacity_percent': read_capacity_percent,
            'recommended_read_split': min(80, read_capacity_percent),  # Max 80% reads
            'reasoning': f"Based on {database_size_gb}GB database, {performance_req} performance, {environment} environment"
        }
    
    def _recommend_deployment_type(self, database_size_gb: int, performance_req: str,
                                 database_engine: str, environment: str) -> Dict:
        """AI-enhanced deployment type recommendation"""
        
        rds_score = 0
        ec2_score = 0
        
        # Size-based scoring
        if database_size_gb < 2000:
            rds_score += 40
        elif database_size_gb > 10000:
            ec2_score += 30
        
        # Performance scoring
        if performance_req == 'high':
            ec2_score += 30
            rds_score += 15
        else:
            rds_score += 35
        
        # Database engine scoring
        if database_engine in ['mysql', 'postgresql']:
            rds_score += 25
        elif database_engine == 'oracle':
            ec2_score += 25  # Complex licensing
        
        # Environment scoring
        if environment == 'production':
            rds_score += 20  # Managed service benefits
        else:
            ec2_score += 10  # Cost savings for non-prod
        
        # Management complexity
        rds_score += 20  # Always prefer managed service
        
        recommendation = 'rds' if rds_score > ec2_score else 'ec2'
        confidence = abs(rds_score - ec2_score) / max(rds_score, ec2_score, 1)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rds_score': rds_score,
            'ec2_score': ec2_score,
            'primary_reasons': self._get_deployment_reasons(recommendation, rds_score, ec2_score)
        }
    
    def _get_deployment_reasons(self, recommendation: str, rds_score: int, ec2_score: int) -> List[str]:
        """Get human-readable reasons for deployment recommendation"""
        
        if recommendation == 'rds':
            return [
                "Managed service reduces operational overhead",
                "Automated backups and patching",
                "Built-in monitoring and alerting",
                "Easy scaling and Multi-AZ deployment",
                "Cost-effective for standard workloads"
            ]
        else:
            return [
                "Maximum performance control and customization",
                "Complex database configurations supported",
                "Potential cost savings for large workloads",
                "Full control over database tuning",
                "Custom backup and disaster recovery strategies"
            ]

class OnPremPerformanceAnalyzer:
    """Analyze on-premises database performance"""
    
    def __init__(self):
        self.cpu_architectures = {
            'intel_xeon_e5': {'base_performance': 1.0, 'single_thread': 0.9, 'multi_thread': 1.1},
            'intel_xeon_sp': {'base_performance': 1.2, 'single_thread': 1.1, 'multi_thread': 1.3},
            'amd_epyc': {'base_performance': 1.15, 'single_thread': 1.0, 'multi_thread': 1.4}
        }
        
        self.storage_types = {
            'sas_hdd': {'iops': 150, 'throughput_mbps': 200, 'latency_ms': 8},
            'sata_ssd': {'iops': 75000, 'throughput_mbps': 500, 'latency_ms': 0.2},
            'nvme_ssd': {'iops': 450000, 'throughput_mbps': 3500, 'latency_ms': 0.05}
        }
    
    def calculate_onprem_performance(self, config: Dict, os_manager: OSPerformanceManager) -> Dict:
        """Calculate comprehensive on-premises performance metrics"""
        
        # Get OS impact
        os_impact = os_manager.calculate_os_performance_impact(
            config['operating_system'], 
            config['server_type'], 
            config['database_engine']
        )
        
        # Calculate CPU performance
        cpu_performance = self._calculate_cpu_performance(config, os_impact)
        
        # Calculate memory performance
        memory_performance = self._calculate_memory_performance(config, os_impact)
        
        # Calculate storage performance
        storage_performance = self._calculate_storage_performance(config, os_impact)
        
        # Calculate network performance
        network_performance = self._calculate_network_performance(config, os_impact)
        
        # Calculate database-specific performance
        database_performance = self._calculate_database_performance(config, os_impact)
        
        # Overall system performance
        overall_performance = self._calculate_overall_performance(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, database_performance, os_impact
        )
        
        return {
            'cpu_performance': cpu_performance,
            'memory_performance': memory_performance,
            'storage_performance': storage_performance,
            'network_performance': network_performance,
            'database_performance': database_performance,
            'overall_performance': overall_performance,
            'os_impact': os_impact,
            'bottlenecks': self._identify_bottlenecks(cpu_performance, memory_performance, 
                                                    storage_performance, network_performance),
            'performance_score': overall_performance['composite_score']
        }
    
    def _calculate_cpu_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate CPU performance metrics"""
        
        # Base CPU calculation
        base_performance = config['cpu_cores'] * config['cpu_ghz']
        
        # OS efficiency impact
        os_adjusted = base_performance * os_impact['cpu_efficiency']
        
        # Platform impact
        if config['server_type'] == 'vmware':
            virtualization_penalty = 1 - os_impact['virtualization_overhead']
            final_performance = os_adjusted * virtualization_penalty
        else:
            final_performance = os_adjusted * 1.05  # Physical boost
        
        # Performance categories
        single_thread_perf = config['cpu_ghz'] * os_impact['cpu_efficiency']
        multi_thread_perf = final_performance
        
        return {
            'base_performance': base_performance,
            'os_adjusted_performance': os_adjusted,
            'final_performance': final_performance,
            'single_thread_performance': single_thread_perf,
            'multi_thread_performance': multi_thread_perf,
            'utilization_estimate': 0.7,  # Typical database utilization
            'efficiency_factor': os_impact['cpu_efficiency']
        }
    
    def _calculate_memory_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate memory performance metrics"""
        
        base_memory = config['ram_gb']
        
        # OS overhead
        if 'windows' in config['operating_system']:
            os_overhead = 4  # GB for Windows
        else:
            os_overhead = 2  # GB for Linux
        
        available_memory = base_memory - os_overhead
        
        # Database memory allocation
        db_memory = available_memory * 0.8  # 80% for database
        buffer_pool = db_memory * 0.7  # 70% for buffer pool
        
        # Memory efficiency
        memory_efficiency = os_impact['memory_efficiency']
        effective_memory = available_memory * memory_efficiency
        
        return {
            'total_memory_gb': base_memory,
            'os_overhead_gb': os_overhead,
            'available_memory_gb': available_memory,
            'database_memory_gb': db_memory,
            'buffer_pool_gb': buffer_pool,
            'effective_memory_gb': effective_memory,
            'memory_efficiency': memory_efficiency,
            'memory_pressure': 'low' if available_memory > 32 else 'medium' if available_memory > 16 else 'high'
        }
    
    def _calculate_storage_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate storage performance metrics"""
        
        # Assume NVMe SSD for modern systems, SAS HDD for older
        if config['cpu_cores'] >= 8:
            storage_type = 'nvme_ssd'
        elif config['cpu_cores'] >= 4:
            storage_type = 'sata_ssd'
        else:
            storage_type = 'sas_hdd'
        
        storage_specs = self.storage_types[storage_type]
        
        # Apply OS I/O efficiency
        effective_iops = storage_specs['iops'] * os_impact['io_efficiency']
        effective_throughput = storage_specs['throughput_mbps'] * os_impact['io_efficiency']
        effective_latency = storage_specs['latency_ms'] / os_impact['io_efficiency']
        
        return {
            'storage_type': storage_type,
            'base_iops': storage_specs['iops'],
            'effective_iops': effective_iops,
            'base_throughput_mbps': storage_specs['throughput_mbps'],
            'effective_throughput_mbps': effective_throughput,
            'base_latency_ms': storage_specs['latency_ms'],
            'effective_latency_ms': effective_latency,
            'io_efficiency': os_impact['io_efficiency']
        }
    
    def _calculate_network_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate network performance metrics"""
        
        base_bandwidth = config['nic_speed']
        
        # OS network efficiency
        effective_bandwidth = base_bandwidth * os_impact['network_efficiency']
        
        # Platform impact
        if config['server_type'] == 'vmware':
            # VMware has additional network virtualization overhead
            effective_bandwidth *= 0.92
        
        return {
            'nic_type': config['nic_type'],
            'base_bandwidth_mbps': base_bandwidth,
            'effective_bandwidth_mbps': effective_bandwidth,
            'network_efficiency': os_impact['network_efficiency'],
            'estimated_latency_ms': 0.1 if 'fiber' in config['nic_type'] else 0.2
        }
    
    def _calculate_database_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate database-specific performance metrics"""
        
        db_optimization = os_impact['db_optimization']
        
        # Calculate database-specific metrics
        if config['database_engine'] == 'mysql':
            base_tps = 5000  # Transactions per second baseline
            connection_limit = 1000
        elif config['database_engine'] == 'postgresql':
            base_tps = 4500
            connection_limit = 500
        elif config['database_engine'] == 'oracle':
            base_tps = 6000
            connection_limit = 2000
        elif config['database_engine'] == 'sqlserver':
            base_tps = 5500
            connection_limit = 1500
        else:  # mongodb
            base_tps = 4000
            connection_limit = 800
        
        # Scale by hardware
        hardware_factor = min(2.0, (config['cpu_cores'] / 4) * (config['ram_gb'] / 16))
        
        # Apply OS optimization
        effective_tps = base_tps * hardware_factor * db_optimization
        
        return {
            'database_engine': config['database_engine'],
            'base_tps': base_tps,
            'hardware_factor': hardware_factor,
            'db_optimization': db_optimization,
            'effective_tps': effective_tps,
            'max_connections': connection_limit,
            'query_cache_efficiency': db_optimization * 0.9
        }
    
    def _calculate_overall_performance(self, cpu_perf: Dict, mem_perf: Dict, 
                                     storage_perf: Dict, net_perf: Dict, 
                                     db_perf: Dict, os_impact: Dict) -> Dict:
        """Calculate overall system performance"""
        
        # Weighted performance score
        cpu_score = min(100, (cpu_perf['final_performance'] / 50) * 100)
        memory_score = min(100, (mem_perf['effective_memory_gb'] / 64) * 100)
        storage_score = min(100, (storage_perf['effective_iops'] / 100000) * 100)
        network_score = min(100, (net_perf['effective_bandwidth_mbps'] / 10000) * 100)
        database_score = min(100, (db_perf['effective_tps'] / 10000) * 100)
        
        # Composite score with weights
        composite_score = (
            cpu_score * 0.25 +
            memory_score * 0.2 +
            storage_score * 0.25 +
            network_score * 0.15 +
            database_score * 0.15
        )
        
        return {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score,
            'composite_score': composite_score,
            'performance_tier': self._get_performance_tier(composite_score),
            'scaling_recommendation': self._get_scaling_recommendation(cpu_score, memory_score, storage_score)
        }
    
    def _get_performance_tier(self, score: float) -> str:
        """Get performance tier based on score"""
        if score >= 80:
            return "High Performance"
        elif score >= 60:
            return "Standard Performance"
        elif score >= 40:
            return "Basic Performance"
        else:
            return "Limited Performance"
    
    def _get_scaling_recommendation(self, cpu_score: float, memory_score: float, storage_score: float) -> List[str]:
        """Get scaling recommendations"""
        recommendations = []
        
        if cpu_score < 60:
            recommendations.append("Consider CPU upgrade or more cores")
        if memory_score < 60:
            recommendations.append("Consider memory expansion")
        if storage_score < 60:
            recommendations.append("Consider storage upgrade to NVMe SSD")
        
        if not recommendations:
            recommendations.append("System is well-balanced for current workload")
        
        return recommendations
    
    def _identify_bottlenecks(self, cpu_perf: Dict, mem_perf: Dict, 
                            storage_perf: Dict, net_perf: Dict) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        if cpu_perf['utilization_estimate'] > 0.8:
            bottlenecks.append("CPU utilization high")
        if mem_perf['memory_pressure'] == 'high':
            bottlenecks.append("Memory pressure detected")
        if storage_perf['effective_latency_ms'] > 5:
            bottlenecks.append("Storage latency high")
        if net_perf['effective_bandwidth_mbps'] < 1000:
            bottlenecks.append("Network bandwidth limited")
        
        return bottlenecks if bottlenecks else ["No significant bottlenecks detected"]

class AWSPerformanceAnalyzer:
    """Analyze AWS post-migration performance"""
    
    def __init__(self):
        self.aws_instance_performance = {
            # CPU performance multipliers based on AWS instance types
            't3': {'cpu_factor': 0.8, 'burst_capable': True, 'baseline_perf': 0.4},
            'c5': {'cpu_factor': 1.2, 'burst_capable': False, 'baseline_perf': 1.0},
            'r6g': {'cpu_factor': 1.1, 'burst_capable': False, 'baseline_perf': 1.0},
            'r6i': {'cpu_factor': 1.15, 'burst_capable': False, 'baseline_perf': 1.0}
        }
        
        self.aws_network_performance = {
            'up_to_10_gbps': {'bandwidth': 10000, 'pps': 1000000},
            '25_gbps': {'bandwidth': 25000, 'pps': 2500000},
            '50_gbps': {'bandwidth': 50000, 'pps': 5000000},
            '100_gbps': {'bandwidth': 100000, 'pps': 10000000}
        }
    
    def calculate_aws_performance(self, config: Dict, onprem_performance: Dict, 
                                aws_sizing: Dict) -> Dict:
        """Calculate expected AWS performance post-migration"""
        
        deployment_type = aws_sizing['deployment_recommendation']['recommendation']
        
        if deployment_type == 'rds':
            aws_performance = self._calculate_rds_performance(config, aws_sizing['rds_recommendations'])
        else:
            aws_performance = self._calculate_ec2_performance(config, aws_sizing['ec2_recommendations'])
        
        # Performance comparison
        performance_comparison = self._compare_performance(onprem_performance, aws_performance)
        
        # Scaling capabilities
        scaling_capabilities = self._calculate_scaling_capabilities(aws_sizing, deployment_type)
        
        # Network latency impact
        network_impact = self._calculate_network_impact(config)
        
        return {
            'aws_performance': aws_performance,
            'performance_comparison': performance_comparison,
            'scaling_capabilities': scaling_capabilities,
            'network_impact': network_impact,
            'deployment_type': deployment_type,
            'optimization_recommendations': self._get_optimization_recommendations(
                aws_performance, performance_comparison
            )
        }
    
    def _calculate_rds_performance(self, config: Dict, rds_config: Dict) -> Dict:
        """Calculate RDS performance metrics"""
        
        instance_specs = rds_config['instance_specs']
        
        # RDS has optimized database performance
        rds_optimization_factor = 1.2  # AWS-optimized database engines
        
        # Calculate performance metrics
        if config['database_engine'] == 'mysql':
            base_tps = 8000
        elif config['database_engine'] == 'postgresql':
            base_tps = 7500
        elif config['database_engine'] == 'oracle':
            base_tps = 9000
        elif config['database_engine'] == 'sqlserver':
            base_tps = 8500
        else:  # DocumentDB
            base_tps = 6000
        
        # Scale by instance size
        hardware_factor = (instance_specs['vcpu'] / 2) * (instance_specs['memory'] / 8)
        effective_tps = base_tps * hardware_factor * rds_optimization_factor
        
        # Storage performance
        if rds_config['storage_type'] == 'gp3':
            storage_iops = min(16000, max(3000, rds_config['storage_size_gb'] * 3))
            storage_throughput = min(1000, max(125, rds_config['storage_size_gb'] * 0.25))
        else:  # io1
            storage_iops = min(64000, rds_config['storage_size_gb'] * 50)
            storage_throughput = min(4000, storage_iops * 0.256)
        
        return {
            'service_type': 'Amazon RDS',
            'instance_type': rds_config['primary_instance'],
            'vcpu': instance_specs['vcpu'],
            'memory_gb': instance_specs['memory'],
            'effective_tps': effective_tps,
            'storage_iops': storage_iops,
            'storage_throughput_mbps': storage_throughput,
            'multi_az': rds_config['multi_az'],
            'automated_backups': True,
            'point_in_time_recovery': True,
            'performance_insights': True,
            'managed_service_benefits': [
                "Automated patching and updates",
                "Built-in monitoring and alerting",
                "Automatic failover (Multi-AZ)",
                "Read replicas for scaling",
                "Performance Insights"
            ]
        }
    
    def _calculate_ec2_performance(self, config: Dict, ec2_config: Dict) -> Dict:
        """Calculate EC2 performance metrics"""
        
        instance_specs = ec2_config['instance_specs']
        
        # EC2 requires more overhead but offers more control
        ec2_overhead_factor = 0.9  # OS and management overhead
        
        # Calculate performance based on instance family
        instance_family = ec2_config['primary_instance'].split('.')[0]
        
        if instance_family in self.aws_instance_performance:
            cpu_factor = self.aws_instance_performance[instance_family]['cpu_factor']
        else:
            cpu_factor = 1.0
        
        # Database performance calculation
        if config['database_engine'] == 'mysql':
            base_tps = 6000
        elif config['database_engine'] == 'postgresql':
            base_tps = 5500
        elif config['database_engine'] == 'oracle':
            base_tps = 7000
        elif config['database_engine'] == 'sqlserver':
            base_tps = 6500
        else:  # mongodb
            base_tps = 5000
        
        hardware_factor = (instance_specs['vcpu'] / 2) * (instance_specs['memory'] / 8)
        effective_tps = base_tps * hardware_factor * cpu_factor * ec2_overhead_factor
        
        # EBS storage performance
        if ec2_config['storage_type'] == 'gp3':
            storage_iops = min(16000, max(3000, ec2_config['storage_size_gb'] * 3))
            storage_throughput = min(1000, max(125, ec2_config['storage_size_gb'] * 0.25))
        else:  # io2
            storage_iops = min(64000, ec2_config['storage_size_gb'] * 500)
            storage_throughput = min(4000, storage_iops * 0.256)
        
        return {
            'service_type': 'Amazon EC2',
            'instance_type': ec2_config['primary_instance'],
            'vcpu': instance_specs['vcpu'],
            'memory_gb': instance_specs['memory'],
            'effective_tps': effective_tps,
            'storage_iops': storage_iops,
            'storage_throughput_mbps': storage_throughput,
            'ebs_optimized': ec2_config['ebs_optimized'],
            'enhanced_networking': ec2_config['enhanced_networking'],
            'self_managed_benefits': [
                "Full control over database configuration",
                "Custom backup strategies",
                "Advanced performance tuning",
                "Flexible maintenance windows",
                "Custom monitoring solutions"
            ]
        }
    
    def _compare_performance(self, onprem: Dict, aws: Dict) -> Dict:
        """Compare on-premises vs AWS performance"""
        
        # TPS comparison
        onprem_tps = onprem['database_performance']['effective_tps']
        aws_tps = aws['aws_performance']['effective_tps']
        tps_improvement = ((aws_tps - onprem_tps) / onprem_tps) * 100
        
        # Storage comparison
        onprem_iops = onprem['storage_performance']['effective_iops']
        aws_iops = aws['aws_performance']['storage_iops']
        iops_improvement = ((aws_iops - onprem_iops) / onprem_iops) * 100
        
        # Memory comparison
        onprem_memory = onprem['memory_performance']['effective_memory_gb']
        aws_memory = aws['aws_performance']['memory_gb']
        memory_improvement = ((aws_memory - onprem_memory) / onprem_memory) * 100
        
        # Overall performance score comparison
        onprem_score = onprem['overall_performance']['composite_score']
        aws_score = min(100, onprem_score * (1 + max(tps_improvement, 0) / 100))
        score_improvement = aws_score - onprem_score
        
        return {
            'tps_comparison': {
                'onprem': onprem_tps,
                'aws': aws_tps,
                'improvement_percent': tps_improvement,
                'status': 'improvement' if tps_improvement > 0 else 'regression' if tps_improvement < -5 else 'similar'
            },
            'iops_comparison': {
                'onprem': onprem_iops,
                'aws': aws_iops,
                'improvement_percent': iops_improvement,
                'status': 'improvement' if iops_improvement > 0 else 'regression' if iops_improvement < -5 else 'similar'
            },
            'memory_comparison': {
                'onprem': onprem_memory,
                'aws': aws_memory,
                'improvement_percent': memory_improvement,
                'status': 'improvement' if memory_improvement > 0 else 'regression' if memory_improvement < -5 else 'similar'
            },
            'overall_score_comparison': {
                'onprem_score': onprem_score,
                'aws_score': aws_score,
                'improvement': score_improvement,
                'status': 'improvement' if score_improvement > 5 else 'regression' if score_improvement < -5 else 'similar'
            }
        }
    
    def _calculate_scaling_capabilities(self, aws_sizing: Dict, deployment_type: str) -> Dict:
        """Calculate AWS scaling capabilities"""
        
        if deployment_type == 'rds':
            scaling_options = {
                'vertical_scaling': {
                    'supported': True,
                    'downtime_required': True,
                    'max_vcpu': 32,
                    'max_memory_gb': 256,
                    'scaling_time_minutes': 5
                },
                'horizontal_scaling': {
                    'read_replicas': True,
                    'max_read_replicas': 5,
                    'cross_region': True,
                    'automatic_failover': True
                },
                'storage_scaling': {
                    'supported': True,
                    'downtime_required': False,
                    'max_size_gb': 65536,
                    'auto_scaling': True
                }
            }
        else:  # EC2
            scaling_options = {
                'vertical_scaling': {
                    'supported': True,
                    'downtime_required': True,
                    'max_vcpu': 32,
                    'max_memory_gb': 256,
                    'scaling_time_minutes': 2
                },
                'horizontal_scaling': {
                    'clustering': True,
                    'load_balancing': True,
                    'auto_scaling_group': True,
                    'cross_az': True
                },
                'storage_scaling': {
                    'supported': True,
                    'downtime_required': False,
                    'max_size_gb': 65536,
                    'snapshot_scaling': True
                }
            }
        
        # Reader/Writer scaling
        rw_config = aws_sizing['reader_writer_config']
        
        return {
            'scaling_options': scaling_options,
            'current_config': {
                'writers': rw_config['writers'],
                'readers': rw_config['readers'],
                'total_instances': rw_config['total_instances']
            },
            'scaling_potential': {
                'max_writers': 2 if deployment_type == 'rds' else 10,
                'max_readers': 15 if deployment_type == 'rds' else 50,
                'geographic_distribution': True,
                'elastic_scaling': deployment_type == 'ec2'
            }
        }
    
    def _calculate_network_impact(self, config: Dict) -> Dict:
        """Calculate network latency impact on performance"""
        
        # Application to database latency in AWS
        aws_internal_latency = 0.5  # ms within AZ
        cross_az_latency = 2.0  # ms cross-AZ
        
        # Compare to on-premises
        onprem_latency = 0.1  # Very low latency on-premises
        
        # Calculate performance impact
        latency_penalty = (aws_internal_latency - onprem_latency) / onprem_latency
        
        return {
            'onprem_latency_ms': onprem_latency,
            'aws_same_az_latency_ms': aws_internal_latency,
            'aws_cross_az_latency_ms': cross_az_latency,
            'latency_impact_percent': latency_penalty * 100,
            'mitigation_strategies': [
                "Deploy applications and database in same AZ",
                "Use connection pooling",
                "Implement application-level caching",
                "Optimize query patterns",
                "Consider read replicas for read-heavy workloads"
            ]
        }
    
    def _get_optimization_recommendations(self, aws_performance: Dict, 
                                        performance_comparison: Dict) -> List[str]:
        """Get AWS performance optimization recommendations"""
        
        recommendations = []
        
        # Instance optimization
        if aws_performance['service_type'] == 'Amazon RDS':
            recommendations.extend([
                "Enable Performance Insights for detailed monitoring",
                "Configure appropriate parameter groups",
                "Set up CloudWatch enhanced monitoring",
                "Consider Aurora for even better performance"
            ])
        else:
            recommendations.extend([
                "Optimize database configuration parameters",
                "Set up detailed CloudWatch metrics",
                "Consider using Placement Groups for low latency",
                "Implement database connection pooling"
            ])
        
        # Storage optimization
        if performance_comparison['iops_comparison']['improvement_percent'] < 50:
            recommendations.append("Consider upgrading to io1/io2 storage for higher IOPS")
        
        # Memory optimization
        if performance_comparison['memory_comparison']['improvement_percent'] < 0:
            recommendations.append("Consider memory-optimized instance types")
        
        # General AWS optimizations
        recommendations.extend([
            "Implement Multi-AZ for high availability",
            "Set up automated backups and snapshots",
            "Use AWS Config for compliance monitoring",
            "Consider Reserved Instances for cost optimization"
        ])
        
        return recommendations

def render_enhanced_header():
    """Enhanced header with v2.0 features"""
    st.markdown("""
    <div class="main-header">
        <h1>☁️ AWS Enterprise Database Migration Analyzer v2.0</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">AI-Powered Sizing • Real Network Paths • Agent Configuration • Live AWS Pricing</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Linux NAS/Windows Share • DataSync/DMS Agents • Production/Non-Production Paths</p>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar_controls():
    """Enhanced sidebar with new network path options"""
    st.sidebar.header("☁️ AWS Migration Configuration v2.0")
    
    # Operating System Selection
    st.sidebar.subheader("💻 Operating System")
    operating_system = st.sidebar.selectbox(
        "OS Selection",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3,
        format_func=lambda x: {
            'windows_server_2019': 'Windows Server 2019',
            'windows_server_2022': 'Windows Server 2022',
            'rhel_8': 'Red Hat Enterprise Linux 8',
            'rhel_9': 'Red Hat Enterprise Linux 9',
            'ubuntu_20_04': 'Ubuntu Server 20.04 LTS',
            'ubuntu_22_04': 'Ubuntu Server 22.04 LTS'
        }[x]
    )
    
    # Platform Configuration
    st.sidebar.subheader("🖥️ Server Platform")
    server_type = st.sidebar.selectbox(
        "Platform Type",
        ["physical", "vmware"],
        format_func=lambda x: "Physical Server" if x == "physical" else "VMware Virtual Machine"
    )
    
    # Hardware Configuration
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256, 512], index=2)
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32, 48, 64], index=2)
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
    # Network Interface
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"],
        index=3
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000, 
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    # Migration Configuration
    st.sidebar.subheader("🔄 Migration Setup")
    
    # Source and Target Databases
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': 'MySQL', 'postgresql': 'PostgreSQL', 'oracle': 'Oracle',
            'sqlserver': 'SQL Server', 'mongodb': 'MongoDB'
        }[x]
    )
    
    database_engine = st.sidebar.selectbox(
        "Target Database (AWS)",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': 'RDS MySQL', 'postgresql': 'RDS PostgreSQL', 'oracle': 'RDS Oracle',
            'sqlserver': 'RDS SQL Server', 'mongodb': 'DocumentDB'
        }[x]
    )
    
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100)
    
    # Migration Parameters
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60)
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"])
    
    # Enhanced Network Path Selection
    st.sidebar.subheader("🌐 Enterprise Network Path")
    network_path = st.sidebar.selectbox(
        "Migration Path",
        ["nonprod_sj_linux_nas", "nonprod_sj_windows_share", "prod_sa_linux_nas", "prod_sa_windows_share"],
        format_func=lambda x: {
            'nonprod_sj_linux_nas': 'Non-Prod: SJ Linux NAS → AWS S3 (2Gbps DX)',
            'nonprod_sj_windows_share': 'Non-Prod: SJ Windows Share → AWS S3 (2Gbps DX)',
            'prod_sa_linux_nas': 'Prod: SA Linux NAS → SJ → AWS VPC S3 (10Gbps)',
            'prod_sa_windows_share': 'Prod: SA Windows Share → SJ → AWS VPC S3 (10Gbps)'
        }[x]
    )
    
    # Environment
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
    # Agent Sizing Selection
    st.sidebar.subheader("🤖 Migration Agent Sizing")
    
    # Determine migration type for tool selection
    is_homogeneous = source_database_engine == database_engine
    primary_tool = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.info(f"**Migration Type:** {'Homogeneous' if is_homogeneous else 'Heterogeneous'}")
    st.sidebar.info(f"**Primary Tool:** AWS {primary_tool}")
    
    if is_homogeneous:
        datasync_agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': 'Small (t3.medium) - Up to 1TB',
                'medium': 'Medium (c5.large) - 1-5TB',
                'large': 'Large (c5.xlarge) - 5-20TB',
                'xlarge': 'XLarge (c5.2xlarge) - >20TB'
            }[x]
        )
        dms_agent_size = None
    else:
        dms_agent_size = st.sidebar.selectbox(
            "DMS Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: {
                'small': 'Small (t3.medium) - Up to 500GB',
                'medium': 'Medium (c5.large) - 500GB-2TB',
                'large': 'Large (c5.xlarge) - 2-10TB',
                'xlarge': 'XLarge (c5.2xlarge) - 10-50TB',
                'xxlarge': 'XXLarge (c5.4xlarge) - >50TB'
            }[x]
        )
        datasync_agent_size = None
    
    if st.sidebar.button("🔄 Refresh Analysis"):
        st.rerun()
    
    return {
        'operating_system': operating_system,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'cpu_ghz': cpu_ghz,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'source_database_engine': source_database_engine,
        'database_engine': database_engine,
        'database_size_gb': database_size_gb,
        'downtime_tolerance_minutes': downtime_tolerance_minutes,
        'performance_requirements': performance_requirements,
        'network_path': network_path,
        'environment': environment,
        'datasync_agent_size': datasync_agent_size,
        'dms_agent_size': dms_agent_size
    }

def render_onprem_performance_tab(onprem_analysis: Dict, config: Dict):
    """Render on-premises performance analysis tab"""
    st.subheader("🖥️ On-Premises Performance Analysis")
    
    # Performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Score", 
            f"{onprem_analysis['performance_score']:.1f}/100",
            delta=onprem_analysis['overall_performance']['performance_tier']
        )
    
    with col2:
        st.metric(
            "Database TPS",
            f"{onprem_analysis['database_performance']['effective_tps']:.0f}",
            delta=f"OS Opt: {onprem_analysis['database_performance']['db_optimization']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Storage IOPS",
            f"{onprem_analysis['storage_performance']['effective_iops']:,.0f}",
            delta=f"{onprem_analysis['storage_performance']['storage_type'].upper()}"
        )
    
    with col4:
        st.metric(
            "Network Bandwidth",
            f"{onprem_analysis['network_performance']['effective_bandwidth_mbps']:,.0f} Mbps",
            delta=f"{config['nic_type'].replace('_', ' ').title()}"
        )
    
    # Detailed performance breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💻 System Configuration & Performance:**")
        st.markdown(f"""
        <div class="onprem-performance-card">
            <h4>🔧 Hardware Configuration</h4>
            <p><strong>OS:</strong> {onprem_analysis['os_impact']['name'] if 'name' in onprem_analysis['os_impact'] else config['operating_system']}</p>
            <p><strong>Platform:</strong> {config['server_type'].title()}</p>
            <p><strong>CPU:</strong> {config['cpu_cores']} cores @ {config['cpu_ghz']} GHz</p>
            <p><strong>Memory:</strong> {config['ram_gb']} GB ({onprem_analysis['memory_performance']['available_memory_gb']:.1f} GB available)</p>
            <p><strong>Storage:</strong> {onprem_analysis['storage_performance']['storage_type'].upper().replace('_', ' ')}</p>
            <p><strong>Network:</strong> {config['nic_type'].replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="onprem-performance-card">
            <h4>📊 Performance Scores</h4>
            <p><strong>CPU Score:</strong> {onprem_analysis['overall_performance']['cpu_score']:.1f}/100</p>
            <p><strong>Memory Score:</strong> {onprem_analysis['overall_performance']['memory_score']:.1f}/100</p>
            <p><strong>Storage Score:</strong> {onprem_analysis['overall_performance']['storage_score']:.1f}/100</p>
            <p><strong>Network Score:</strong> {onprem_analysis['overall_performance']['network_score']:.1f}/100</p>
            <p><strong>Database Score:</strong> {onprem_analysis['overall_performance']['database_score']:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🎯 OS Performance Impact:**")
        st.markdown(f"""
        <div class="os-comparison-card">
            <h4>⚙️ Operating System Efficiency</h4>
            <p><strong>Overall Efficiency:</strong> {onprem_analysis['os_impact']['total_efficiency']*100:.1f}%</p>
            <p><strong>CPU Efficiency:</strong> {onprem_analysis['os_impact']['cpu_efficiency']*100:.1f}%</p>
            <p><strong>Memory Efficiency:</strong> {onprem_analysis['os_impact']['memory_efficiency']*100:.1f}%</p>
            <p><strong>I/O Efficiency:</strong> {onprem_analysis['os_impact']['io_efficiency']*100:.1f}%</p>
            <p><strong>Network Efficiency:</strong> {onprem_analysis['os_impact']['network_efficiency']*100:.1f}%</p>
            <p><strong>DB Optimization:</strong> {onprem_analysis['os_impact']['db_optimization']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="onprem-performance-card">
            <h4>⚠️ Bottlenecks & Issues</h4>
            <ul>
                {"".join([f"<li>{bottleneck}</li>" for bottleneck in onprem_analysis['bottlenecks']])}
            </ul>
            <h5>📈 Scaling Recommendations:</h5>
            <ul>
                {"".join([f"<li>{rec}</li>" for rec in onprem_analysis['overall_performance']['scaling_recommendation']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance comparison charts
    st.markdown("**📊 Performance Metrics Visualization:**")
    
    # Create performance comparison data
    performance_data = {
        'Component': ['CPU', 'Memory', 'Storage', 'Network', 'Database'],
        'Score': [
            onprem_analysis['overall_performance']['cpu_score'],
            onprem_analysis['overall_performance']['memory_score'],
            onprem_analysis['overall_performance']['storage_score'],
            onprem_analysis['overall_performance']['network_score'],
            onprem_analysis['overall_performance']['database_score']
        ],
        'Category': ['Hardware', 'Hardware', 'Hardware', 'Hardware', 'Software']
    }
    
    fig_performance = px.bar(
        performance_data, 
        x='Component', 
        y='Score',
        color='Category',
        title="On-Premises Performance Component Analysis",
        color_discrete_map={'Hardware': '#3498db', 'Software': '#e74c3c'}
    )
    fig_performance.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # OS Efficiency radar chart
    st.markdown("**🎯 Operating System Efficiency Analysis:**")
    
    os_metrics = ['CPU', 'Memory', 'I/O', 'Network', 'DB Optimization']
    os_values = [
        onprem_analysis['os_impact']['cpu_efficiency'] * 100,
        onprem_analysis['os_impact']['memory_efficiency'] * 100,
        onprem_analysis['os_impact']['io_efficiency'] * 100,
        onprem_analysis['os_impact']['network_efficiency'] * 100,
        onprem_analysis['os_impact']['db_optimization'] * 100
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=os_values,
        theta=os_metrics,
        fill='toself',
        name='OS Efficiency',
        line_color='#2ecc71'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Operating System Performance Efficiency"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

def render_aws_performance_tab(aws_analysis: Dict, config: Dict):
    """Render AWS post-migration performance analysis tab"""
    st.subheader("☁️ AWS Post-Migration Performance Analysis")
    
    aws_perf = aws_analysis['aws_performance']
    comparison = aws_analysis['performance_comparison']
    
    # Performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        tps_delta = comparison['tps_comparison']['improvement_percent']
        delta_class = "delta-positive" if tps_delta > 0 else "delta-negative" if tps_delta < -5 else "delta-neutral"
        st.metric(
            "Database TPS",
            f"{aws_perf['effective_tps']:.0f}",
            delta=f"{tps_delta:+.1f}%"
        )
    
    with col2:
        iops_delta = comparison['iops_comparison']['improvement_percent']
        st.metric(
            "Storage IOPS",
            f"{aws_perf['storage_iops']:,.0f}",
            delta=f"{iops_delta:+.1f}%"
        )
    
    with col3:
        memory_delta = comparison['memory_comparison']['improvement_percent']
        st.metric(
            "Memory",
            f"{aws_perf['memory_gb']} GB",
            delta=f"{memory_delta:+.1f}%"
        )
    
    with col4:
        overall_delta = comparison['overall_score_comparison']['improvement']
        st.metric(
            "Performance Score",
            f"{comparison['overall_score_comparison']['aws_score']:.1f}/100",
            delta=f"{overall_delta:+.1f} points"
        )
    
    # AWS service details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="aws-performance-card">
            <h4>☁️ AWS Service Configuration</h4>
            <p><strong>Service:</strong> {aws_perf['service_type']}</p>
            <p><strong>Instance:</strong> {aws_perf['instance_type']}</p>
            <p><strong>vCPU:</strong> {aws_perf['vcpu']} cores</p>
            <p><strong>Memory:</strong> {aws_perf['memory_gb']} GB</p>
            <p><strong>Storage IOPS:</strong> {aws_perf['storage_iops']:,}</p>
            <p><strong>Storage Throughput:</strong> {aws_perf['storage_throughput_mbps']:.0f} MB/s</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Service-specific benefits
        if aws_perf['service_type'] == 'Amazon RDS':
            benefits = aws_perf['managed_service_benefits']
            st.markdown(f"""
            <div class="rds-recommendation">
                <h4>🎯 RDS Managed Service Benefits</h4>
                <ul>
                    {"".join([f"<li>{benefit}</li>" for benefit in benefits])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            benefits = aws_perf['self_managed_benefits']
            st.markdown(f"""
            <div class="ec2-recommendation">
                <h4>🔧 EC2 Self-Managed Benefits</h4>
                <ul>
                    {"".join([f"<li>{benefit}</li>" for benefit in benefits])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📈 Performance Comparison Summary:**")
        
        # Performance comparison cards
        for metric, data in comparison.items():
            if metric.endswith('_comparison'):
                metric_name = metric.replace('_comparison', '').replace('_', ' ').title()
                if metric_name == 'Overall Score':
                    improvement = data['improvement']
                    status_icon = "📈" if improvement > 5 else "📉" if improvement < -5 else "➡️"
                    st.markdown(f"""
                    <div class="performance-card">
                        <h5>{status_icon} {metric_name}</h5>
                        <p><strong>On-Prem:</strong> {data['onprem_score']:.1f}</p>
                        <p><strong>AWS:</strong> {data['aws_score']:.1f}</p>
                        <p><strong>Change:</strong> {improvement:+.1f} points</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    improvement = data['improvement_percent']
                    status_icon = "📈" if improvement > 0 else "📉" if improvement < -5 else "➡️"
                    st.markdown(f"""
                    <div class="performance-card">
                        <h5>{status_icon} {metric_name}</h5>
                        <p><strong>On-Prem:</strong> {data['onprem']:,.0f}</p>
                        <p><strong>AWS:</strong> {data['aws']:,.0f}</p>
                        <p><strong>Change:</strong> {improvement:+.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Scaling capabilities
    st.markdown("**🚀 AWS Scaling Capabilities:**")
    
    scaling = aws_analysis['scaling_capabilities']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vertical = scaling['scaling_options']['vertical_scaling']
        st.markdown(f"""
        <div class="aws-performance-card">
            <h4>📊 Vertical Scaling</h4>
            <p><strong>Supported:</strong> {'✅' if vertical['supported'] else '❌'}</p>
            <p><strong>Max vCPU:</strong> {vertical['max_vcpu']}</p>
            <p><strong>Max Memory:</strong> {vertical['max_memory_gb']} GB</p>
            <p><strong>Downtime:</strong> {'Required' if vertical['downtime_required'] else 'Not Required'}</p>
            <p><strong>Scaling Time:</strong> {vertical['scaling_time_minutes']} minutes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        horizontal = scaling['scaling_options']['horizontal_scaling']
        st.markdown(f"""
        <div class="aws-performance-card">
            <h4>📈 Horizontal Scaling</h4>
            {f"<p><strong>Read Replicas:</strong> {'✅' if horizontal.get('read_replicas') else '❌'}</p>" if 'read_replicas' in horizontal else ""}
            {f"<p><strong>Max Replicas:</strong> {horizontal.get('max_read_replicas', 'N/A')}</p>" if 'max_read_replicas' in horizontal else ""}
            {f"<p><strong>Clustering:</strong> {'✅' if horizontal.get('clustering') else '❌'}</p>" if 'clustering' in horizontal else ""}
            {f"<p><strong>Auto Scaling:</strong> {'✅' if horizontal.get('auto_scaling_group') else '❌'}</p>" if 'auto_scaling_group' in horizontal else ""}
            {f"<p><strong>Cross-Region:</strong> {'✅' if horizontal.get('cross_region') else '❌'}</p>" if 'cross_region' in horizontal else ""}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        storage = scaling['scaling_options']['storage_scaling']
        st.markdown(f"""
        <div class="aws-performance-card">
            <h4>💾 Storage Scaling</h4>
            <p><strong>Supported:</strong> {'✅' if storage['supported'] else '❌'}</p>
            <p><strong>Max Size:</strong> {storage['max_size_gb']:,} GB</p>
            <p><strong>Downtime:</strong> {'Required' if storage['downtime_required'] else 'Not Required'}</p>
            {f"<p><strong>Auto Scaling:</strong> {'✅' if storage.get('auto_scaling') else '❌'}</p>" if 'auto_scaling' in storage else ""}
        </div>
        """, unsafe_allow_html=True)
    
    # Network impact analysis
    st.markdown("**🌐 Network Latency Impact Analysis:**")
    
    network_impact = aws_analysis['network_impact']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="network-path-card">
            <h4>📡 Latency Comparison</h4>
            <p><strong>On-Premises:</strong> {network_impact['onprem_latency_ms']:.1f} ms</p>
            <p><strong>AWS Same AZ:</strong> {network_impact['aws_same_az_latency_ms']:.1f} ms</p>
            <p><strong>AWS Cross AZ:</strong> {network_impact['aws_cross_az_latency_ms']:.1f} ms</p>
            <p><strong>Impact:</strong> {network_impact['latency_impact_percent']:+.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="network-path-card">
            <h4>🛠️ Mitigation Strategies</h4>
            <ul>
                {"".join([f"<li>{strategy}</li>" for strategy in network_impact['mitigation_strategies'][:4]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Optimization recommendations
    st.markdown("**💡 AWS Optimization Recommendations:**")
    
    recommendations = aws_analysis['optimization_recommendations']
    
    # Group recommendations by category
    performance_recs = [r for r in recommendations if any(word in r.lower() for word in ['performance', 'monitoring', 'insights'])]
    scaling_recs = [r for r in recommendations if any(word in r.lower() for word in ['scaling', 'instance', 'memory'])]
    operational_recs = [r for r in recommendations if any(word in r.lower() for word in ['backup', 'multi-az', 'config', 'reserved'])]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if performance_recs:
            st.markdown(f"""
            <div class="aws-recommendation-card">
                <h4>🎯 Performance Optimization</h4>
                <ul>
                    {"".join([f"<li>{rec}</li>" for rec in performance_recs])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if scaling_recs:
            st.markdown(f"""
            <div class="aws-recommendation-card">
                <h4>📊 Scaling & Sizing</h4>
                <ul>
                    {"".join([f"<li>{rec}</li>" for rec in scaling_recs])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if operational_recs:
            st.markdown(f"""
            <div class="aws-recommendation-card">
                <h4>🔧 Operational Excellence</h4>
                <ul>
                    {"".join([f"<li>{rec}</li>" for rec in operational_recs])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance comparison visualization
    st.markdown("**📊 Performance Comparison Charts:**")
    
    # Create comparison data
    metrics = ['TPS', 'IOPS', 'Memory (GB)']
    onprem_values = [
        comparison['tps_comparison']['onprem'],
        comparison['iops_comparison']['onprem'],
        comparison['memory_comparison']['onprem']
    ]
    aws_values = [
        comparison['tps_comparison']['aws'],
        comparison['iops_comparison']['aws'],
        comparison['memory_comparison']['aws']
    ]
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        name='On-Premises',
        x=metrics,
        y=onprem_values,
        marker_color='#3498db'
    ))
    fig_comparison.add_trace(go.Bar(
        name='AWS',
        x=metrics,
        y=aws_values,
        marker_color='#e74c3c'
    ))
    
    fig_comparison.update_layout(
        title="On-Premises vs AWS Performance Comparison",
        barmode='group',
        yaxis_title="Performance Units"
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)

def render_enhanced_aws_recommendations(analysis: Dict, config: Dict):
    """Render enhanced AWS sizing recommendations"""
    st.subheader("🎯 AI-Powered AWS Sizing Recommendations")
    
    sizing_recs = analysis['aws_sizing_recommendations']
    
    # Deployment recommendation
    deployment = sizing_recs['deployment_recommendation']
    
    st.markdown(f"""
    <div class="aws-recommendation-card">
        <h4>🚀 Deployment Recommendation: {deployment['recommendation'].upper()}</h4>
        <p><strong>Confidence:</strong> {deployment['confidence']*100:.1f}%</p>
        <p><strong>Primary Reasons:</strong></p>
        <ul>
            {"".join([f"<li>{reason}</li>" for reason in deployment['primary_reasons'][:3]])}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Side-by-side RDS vs EC2 recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        rds_rec = sizing_recs['rds_recommendations']
        st.markdown(f"""
        <div class="rds-recommendation">
            <h4>☁️ RDS Recommendation</h4>
            <p><strong>Instance:</strong> {rds_rec['primary_instance']}</p>
            <p><strong>vCPU:</strong> {rds_rec['instance_specs']['vcpu']} cores</p>
            <p><strong>Memory:</strong> {rds_rec['instance_specs']['memory']} GB</p>
            <p><strong>Storage:</strong> {rds_rec['storage_type'].upper()} ({rds_rec['storage_size_gb']} GB)</p>
            <p><strong>Monthly Cost:</strong> ${rds_rec['total_monthly_cost']:.0f}</p>
            <p><strong>Multi-AZ:</strong> {'Yes' if rds_rec['multi_az'] else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ec2_rec = sizing_recs['ec2_recommendations']
        st.markdown(f"""
        <div class="ec2-recommendation">
            <h4>🖥️ EC2 Recommendation</h4>
            <p><strong>Instance:</strong> {ec2_rec['primary_instance']}</p>
            <p><strong>vCPU:</strong> {ec2_rec['instance_specs']['vcpu']} cores</p>
            <p><strong>Memory:</strong> {ec2_rec['instance_specs']['memory']} GB</p>
            <p><strong>Storage:</strong> {ec2_rec['storage_type'].upper()} ({ec2_rec['storage_size_gb']} GB)</p>
            <p><strong>Monthly Cost:</strong> ${ec2_rec['total_monthly_cost']:.0f}</p>
            <p><strong>EBS Optimized:</strong> {'Yes' if ec2_rec['ebs_optimized'] else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reader/Writer Configuration
    rw_config = sizing_recs['reader_writer_config']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Writer Instances", rw_config['writers'])
    
    with col2:
        st.metric("Reader Instances", rw_config['readers'])
    
    with col3:
        st.metric("Total Instances", rw_config['total_instances'])
    
    st.markdown(f"""
    <div class="aws-recommendation-card">
        <h4>📊 Read/Write Distribution Strategy</h4>
        <p><strong>Write Capacity:</strong> {rw_config['write_capacity_percent']:.1f}%</p>
        <p><strong>Read Capacity:</strong> {rw_config['read_capacity_percent']:.1f}%</p>
        <p><strong>Recommended Read Split:</strong> {rw_config['recommended_read_split']:.1f}%</p>
        <p><strong>Reasoning:</strong> {rw_config['reasoning']}</p>
    </div>
    """, unsafe_allow_html=True)

def render_agent_sizing_analysis(analysis: Dict, config: Dict):
    """Render migration agent sizing analysis"""
    st.subheader("🤖 Migration Agent Sizing & Configuration")
    
    agent_analysis = analysis['agent_analysis']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if agent_analysis['primary_tool'] == 'datasync':
            agent_config = agent_analysis['datasync_config']
            st.markdown(f"""
            <div class="agent-sizing-card">
                <h4>📦 AWS DataSync Agent Configuration</h4>
                <p><strong>Size:</strong> {agent_config['name']}</p>
                <p><strong>vCPU:</strong> {agent_config['vcpu']} cores</p>
                <p><strong>Memory:</strong> {agent_config['memory_gb']} GB</p>
                <p><strong>Max Throughput:</strong> {agent_config['max_throughput_mbps']} Mbps</p>
                <p><strong>Concurrent Tasks:</strong> {agent_config['max_concurrent_tasks']}</p>
                <p><strong>Hourly Cost:</strong> ${agent_config['cost_per_hour']:.4f}</p>
                <p><strong>Recommended For:</strong> {agent_config['recommended_for']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            agent_config = agent_analysis['dms_config']
            st.markdown(f"""
            <div class="agent-sizing-card">
                <h4>🔄 AWS DMS Instance Configuration</h4>
                <p><strong>Size:</strong> {agent_config['name']}</p>
                <p><strong>vCPU:</strong> {agent_config['vcpu']} cores</p>
                <p><strong>Memory:</strong> {agent_config['memory_gb']} GB</p>
                <p><strong>Max Throughput:</strong> {agent_config['max_throughput_mbps']} Mbps</p>
                <p><strong>Concurrent Tasks:</strong> {agent_config['max_concurrent_tasks']}</p>
                <p><strong>Hourly Cost:</strong> ${agent_config['cost_per_hour']:.4f}</p>
                <p><strong>Recommended For:</strong> {agent_config['recommended_for']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Migration performance impact
        throughput_impact = agent_analysis['throughput_impact']
        migration_time = analysis['estimated_migration_time_hours']
        
        st.markdown(f"""
        <div class="agent-sizing-card">
            <h4>📊 Performance Impact Analysis</h4>
            <p><strong>Agent Throughput:</strong> {agent_config['max_throughput_mbps']} Mbps</p>
            <p><strong>Network Bandwidth:</strong> {analysis['network_performance']['effective_bandwidth_mbps']:.0f} Mbps</p>
            <p><strong>Bottleneck:</strong> {'Agent' if agent_config['max_throughput_mbps'] < analysis['network_performance']['effective_bandwidth_mbps'] else 'Network'}</p>
            <p><strong>Effective Throughput:</strong> {min(agent_config['max_throughput_mbps'], analysis['network_performance']['effective_bandwidth_mbps']):.0f} Mbps</p>
            <p><strong>Estimated Migration Time:</strong> {migration_time:.1f} hours</p>
            <p><strong>Agent Utilization:</strong> {throughput_impact*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Agent sizing comparison
    st.markdown("**🔍 Agent Size Comparison:**")
    
    if agent_analysis['primary_tool'] == 'datasync':
        agent_manager = AgentSizingManager()
        agents = agent_manager.datasync_agents
    else:
        agent_manager = AgentSizingManager()
        agents = agent_manager.dms_agents
    
    comparison_data = []
    for size, config in agents.items():
        comparison_data.append({
            'Size': config['name'],
            'vCPU': config['vcpu'],
            'Memory (GB)': config['memory_gb'],
            'Max Throughput (Mbps)': config['max_throughput_mbps'],
            'Concurrent Tasks': config['max_concurrent_tasks'],
            'Hourly Cost': f"${config['cost_per_hour']:.4f}",
            'Monthly Cost': f"${config['cost_per_hour'] * 24 * 30:.0f}",
            'Recommended For': config['recommended_for']
        })
    
    df_agents = pd.DataFrame(comparison_data)
    st.dataframe(df_agents, use_container_width=True, hide_index=True)

def render_enhanced_network_analysis(analysis: Dict, config: Dict):
    """Render enhanced network path analysis with realistic scenarios"""
    st.subheader("🌐 Enterprise Network Path Analysis")
    
    network_perf = analysis['network_performance']
    
    # Network path overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="network-path-card">
            <h4>📍 Path Details</h4>
            <p><strong>Route:</strong> {network_perf['path_name']}</p>
            <p><strong>Environment:</strong> {network_perf['environment'].title()}</p>
            <p><strong>OS Type:</strong> {network_perf['os_type'].title()}</p>
            <p><strong>Storage:</strong> {network_perf['storage_type'].upper()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="network-path-card">
            <h4>📊 Performance</h4>
            <p><strong>Bandwidth:</strong> {network_perf['effective_bandwidth_mbps']:.0f} Mbps</p>
            <p><strong>Latency:</strong> {network_perf['total_latency_ms']:.1f} ms</p>
            <p><strong>Reliability:</strong> {network_perf['total_reliability']*100:.3f}%</p>
            <p><strong>Quality Score:</strong> {network_perf['network_quality_score']:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="network-path-card">
            <h4>🚀 Migration Impact</h4>
            <p><strong>Migration Throughput:</strong> {analysis['migration_throughput_mbps']:.0f} Mbps</p>
            <p><strong>Migration Time:</strong> {analysis['estimated_migration_time_hours']:.1f} hours</p>
            <p><strong>Network Utilization:</strong> {(analysis['migration_throughput_mbps'] / network_perf['effective_bandwidth_mbps']) * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="network-path-card">
            <h4>💰 Cost Impact</h4>
            <p><strong>Cost Factor:</strong> {network_perf['total_cost_factor']:.1f}x</p>
            <p><strong>Segments:</strong> {len(network_perf['segments'])}</p>
            <p><strong>Connection Types:</strong> {len(set(s['connection_type'] for s in network_perf['segments']))}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed segment analysis
    st.markdown("**🔗 Network Segment Performance Details:**")
    
    segment_data = []
    for i, segment in enumerate(network_perf['segments']):
        segment_data.append({
            'Segment': segment['name'],
            'Connection Type': segment['connection_type'].replace('_', ' ').title(),
            'Base Bandwidth (Mbps)': f"{segment['bandwidth_mbps']:,}",
            'Effective Bandwidth (Mbps)': f"{segment['effective_bandwidth_mbps']:.0f}",
            'Base Latency (ms)': f"{segment['latency_ms']:.1f}",
            'Effective Latency (ms)': f"{segment['effective_latency_ms']:.1f}",
            'Reliability': f"{segment['reliability']*100:.2f}%",
            'Cost Factor': f"{segment['cost_factor']:.1f}x"
        })
    
    df_segments = pd.DataFrame(segment_data)
    st.dataframe(df_segments, use_container_width=True, hide_index=True)

def render_enhanced_cost_analysis(analysis: Dict, config: Dict):
    """Render comprehensive cost analysis with AWS pricing"""
    st.subheader("💰 Comprehensive Migration Cost Analysis")
    
    cost_analysis = analysis['cost_analysis']
    
    # High-level cost metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("AWS Compute", f"${cost_analysis['aws_compute_cost']:.0f}/mo")
    
    with col2:
        st.metric("AWS Storage", f"${cost_analysis['aws_storage_cost']:.0f}/mo")
    
    with col3:
        st.metric("Network/DX", f"${cost_analysis['network_cost']:.0f}/mo")
    
    with col4:
        st.metric("Migration Agents", f"${cost_analysis['agent_cost']:.0f}/mo")
    
    with col5:
        st.metric("Total Monthly", f"${cost_analysis['total_monthly_cost']:.0f}")
    
    # Detailed cost breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Monthly Cost Breakdown:**")
        
        cost_breakdown = pd.DataFrame([
            {'Component': 'AWS Compute (Recommended)', 'Monthly Cost': f"${cost_analysis['aws_compute_cost']:.0f}"},
            {'Component': 'AWS Storage', 'Monthly Cost': f"${cost_analysis['aws_storage_cost']:.0f}"},
            {'Component': 'Network & Direct Connect', 'Monthly Cost': f"${cost_analysis['network_cost']:.0f}"},
            {'Component': 'Migration Agents', 'Monthly Cost': f"${cost_analysis['agent_cost']:.0f}"},
            {'Component': 'OS Licensing', 'Monthly Cost': f"${cost_analysis['os_licensing_cost']:.0f}"},
            {'Component': 'Management & Support', 'Monthly Cost': f"${cost_analysis['management_cost']:.0f}"},
            {'Component': 'TOTAL MONTHLY', 'Monthly Cost': f"${cost_analysis['total_monthly_cost']:.0f}"}
        ])
        
        st.dataframe(cost_breakdown, use_container_width=True, hide_index=True)
    
    with col2:
        # RDS vs EC2 cost comparison
        sizing_recs = analysis['aws_sizing_recommendations']
        
        st.markdown("**⚖️ RDS vs EC2 Cost Comparison:**")
        
        comparison_data = pd.DataFrame([
            {
                'Deployment': 'Amazon RDS',
                'Instance Cost': f"${sizing_recs['rds_recommendations']['monthly_instance_cost']:.0f}",
                'Storage Cost': f"${sizing_recs['rds_recommendations']['monthly_storage_cost']:.0f}",
                'Total Monthly': f"${sizing_recs['rds_recommendations']['total_monthly_cost']:.0f}",
                'Annual Total': f"${sizing_recs['rds_recommendations']['total_monthly_cost'] * 12:.0f}"
            },
            {
                'Deployment': 'Amazon EC2',
                'Instance Cost': f"${sizing_recs['ec2_recommendations']['monthly_instance_cost']:.0f}",
                'Storage Cost': f"${sizing_recs['ec2_recommendations']['monthly_storage_cost']:.0f}",
                'Total Monthly': f"${sizing_recs['ec2_recommendations']['total_monthly_cost']:.0f}",
                'Annual Total': f"${sizing_recs['ec2_recommendations']['total_monthly_cost'] * 12:.0f}"
            }
        ])
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # Cost projections
    st.markdown("**📈 Cost Projections & TCO Analysis:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="pricing-card">
            <h4>📅 1-Year Projection</h4>
            <p><strong>Total Cost:</strong> ${cost_analysis['total_monthly_cost'] * 12:.0f}</p>
            <p><strong>Migration Cost:</strong> ${cost_analysis['one_time_migration_cost']:.0f}</p>
            <p><strong>First Year Total:</strong> ${cost_analysis['total_monthly_cost'] * 12 + cost_analysis['one_time_migration_cost']:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="pricing-card">
            <h4>📅 3-Year TCO</h4>
            <p><strong>Operational Cost:</strong> ${cost_analysis['total_monthly_cost'] * 36:.0f}</p>
            <p><strong>Migration Cost:</strong> ${cost_analysis['one_time_migration_cost']:.0f}</p>
            <p><strong>3-Year Total:</strong> ${cost_analysis['total_monthly_cost'] * 36 + cost_analysis['one_time_migration_cost']:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        monthly_savings = cost_analysis.get('estimated_monthly_savings', 0)
        st.markdown(f"""
        <div class="pricing-card">
            <h4>💡 Cost Optimization</h4>
            <p><strong>Potential Monthly Savings:</strong> ${monthly_savings:.0f}</p>
            <p><strong>Annual Savings:</strong> ${monthly_savings * 12:.0f}</p>
            <p><strong>ROI Timeline:</strong> {cost_analysis.get('roi_months', 'N/A')} months</p>
        </div>
        """, unsafe_allow_html=True)

class EnhancedMigrationAnalyzer:
    """Enhanced migration analyzer with all v2.0 features"""
    
    def __init__(self):
        self.os_manager = OSPerformanceManager()
        self.aws_manager = EnhancedAWSMigrationManager()
        self.network_manager = EnhancedNetworkPathManager()
        self.agent_manager = AgentSizingManager()
        self.onprem_analyzer = OnPremPerformanceAnalyzer()
        self.aws_analyzer = AWSPerformanceAnalyzer()
        
        # Enhanced hardware manager (simplified for this implementation)
        self.nic_types = {
            'gigabit_copper': {'max_speed': 1000, 'efficiency': 0.85},
            'gigabit_fiber': {'max_speed': 1000, 'efficiency': 0.90},
            '10g_copper': {'max_speed': 10000, 'efficiency': 0.88},
            '10g_fiber': {'max_speed': 10000, 'efficiency': 0.92},
            '25g_fiber': {'max_speed': 25000, 'efficiency': 0.94},
            '40g_fiber': {'max_speed': 40000, 'efficiency': 0.95}
        }
    
    def comprehensive_migration_analysis(self, config: Dict) -> Dict:
        """Comprehensive v2.0 migration analysis"""
        
        # On-premises performance analysis
        onprem_performance = self.onprem_analyzer.calculate_onprem_performance(config, self.os_manager)
        
        # Hardware performance calculation
        hardware_perf = self._calculate_hardware_performance(config)
        
        # Network path analysis
        network_perf = self.network_manager.calculate_path_performance(config['network_path'])
        
        # Determine migration type and tools
        is_homogeneous = config['source_database_engine'] == config['database_engine']
        migration_type = 'homogeneous' if is_homogeneous else 'heterogeneous'
        primary_tool = 'datasync' if is_homogeneous else 'dms'
        
        # Agent analysis
        agent_analysis = self._analyze_migration_agents(config, primary_tool, network_perf)
        
        # Calculate effective migration throughput
        agent_throughput = agent_analysis['effective_throughput']
        network_throughput = network_perf['effective_bandwidth_mbps']
        migration_throughput = min(agent_throughput, network_throughput)
        
        # Calculate migration time
        database_size_gb = config['database_size_gb']
        migration_time_hours = (database_size_gb * 8 * 1000) / (migration_throughput * 3600)
        
        # AWS sizing recommendations
        aws_sizing = self.aws_manager.recommend_aws_sizing(config)
        
        # AWS performance analysis
        aws_performance_analysis = self.aws_analyzer.calculate_aws_performance(
            config, onprem_performance, aws_sizing
        )
        
        # Cost analysis
        cost_analysis = self._calculate_comprehensive_costs(config, aws_sizing, agent_analysis, network_perf)
        
        return {
            'onprem_performance': onprem_performance,
            'hardware_performance': hardware_perf,
            'network_performance': network_perf,
            'migration_type': migration_type,
            'primary_tool': primary_tool,
            'agent_analysis': agent_analysis,
            'migration_throughput_mbps': migration_throughput,
            'estimated_migration_time_hours': migration_time_hours,
            'aws_sizing_recommendations': aws_sizing,
            'aws_performance_analysis': aws_performance_analysis,
            'cost_analysis': cost_analysis
        }
    
    def _calculate_hardware_performance(self, config: Dict) -> Dict:
        """Calculate hardware performance with OS impact"""
        
        # Get OS impact
        os_impact = self.os_manager.calculate_os_performance_impact(
            config['operating_system'], 
            config['server_type'], 
            config['database_engine']
        )
        
        # Calculate base hardware performance
        nic_config = self.nic_types[config['nic_type']]
        
        # Simple performance calculation
        cpu_performance = min(1.0, (config['cpu_cores'] * config['cpu_ghz']) / 32)
        memory_performance = min(1.0, config['ram_gb'] / 128)
        nic_performance = config['nic_speed'] * nic_config['efficiency']
        
        # Apply OS efficiency
        os_adjusted_performance = nic_performance * os_impact['total_efficiency']
        
        return {
            'cpu_performance': cpu_performance,
            'memory_performance': memory_performance,
            'nic_performance': nic_performance,
            'os_impact': os_impact,
            'os_adjusted_throughput': os_adjusted_performance,
            'os_efficiency_factor': os_impact['total_efficiency'],
            'os_licensing_cost': os_impact['licensing_cost_factor'],
            'os_management_complexity': os_impact['management_complexity']
        }
    
    def _analyze_migration_agents(self, config: Dict, primary_tool: str, network_perf: Dict) -> Dict:
        """Analyze migration agent configuration and performance"""
        
        if primary_tool == 'datasync':
            agent_size = config['datasync_agent_size']
            agent_config = self.agent_manager.datasync_agents[agent_size]
            tool_config = None
        else:
            agent_size = config['dms_agent_size']
            agent_config = self.agent_manager.dms_agents[agent_size]
            tool_config = None
        
        # Calculate throughput impact
        agent_max_throughput = agent_config['max_throughput_mbps']
        network_bandwidth = network_perf['effective_bandwidth_mbps']
        
        # Effective throughput is limited by the bottleneck
        effective_throughput = min(agent_max_throughput, network_bandwidth)
        throughput_impact = effective_throughput / agent_max_throughput
        
        return {
            'primary_tool': primary_tool,
            'agent_size': agent_size,
            f'{primary_tool}_config': agent_config,
            'effective_throughput': effective_throughput,
            'throughput_impact': throughput_impact,
            'bottleneck': 'agent' if agent_max_throughput < network_bandwidth else 'network'
        }
    
    def _calculate_comprehensive_costs(self, config: Dict, aws_sizing: Dict, 
                                     agent_analysis: Dict, network_perf: Dict) -> Dict:
        """Calculate comprehensive costs including AWS pricing"""
        
        # Get recommended deployment cost
        deployment_rec = aws_sizing['deployment_recommendation']['recommendation']
        
        if deployment_rec == 'rds':
            aws_compute_cost = aws_sizing['rds_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['rds_recommendations']['monthly_storage_cost']
        else:
            aws_compute_cost = aws_sizing['ec2_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['ec2_recommendations']['monthly_storage_cost']
        
        # Agent costs
        agent_config = agent_analysis[f"{agent_analysis['primary_tool']}_config"]
        agent_cost = agent_config['cost_per_hour'] * 24 * 30  # Monthly
        
        # Network costs (simplified)
        if 'prod' in config['network_path']:
            network_cost = 800  # Production DX costs
        else:
            network_cost = 400  # Non-production DX costs
        
        # OS licensing
        os_licensing_cost = self.os_manager.operating_systems[config['operating_system']]['licensing_cost_factor'] * 150
        
        # Management costs
        management_cost = 200 if deployment_rec == 'ec2' else 50  # RDS is managed
        
        # One-time migration costs
        complexity_factor = 0.8 if agent_analysis['primary_tool'] == 'dms' else 0.3
        one_time_migration_cost = config['database_size_gb'] * complexity_factor * 0.1
        
        total_monthly_cost = (aws_compute_cost + aws_storage_cost + agent_cost + 
                            network_cost + os_licensing_cost + management_cost)
        
        # Estimated savings (placeholder)
        estimated_monthly_savings = total_monthly_cost * 0.15  # Assume 15% optimization potential
        roi_months = int(one_time_migration_cost / estimated_monthly_savings) if estimated_monthly_savings > 0 else None
        
        return {
            'aws_compute_cost': aws_compute_cost,
            'aws_storage_cost': aws_storage_cost,
            'agent_cost': agent_cost,
            'network_cost': network_cost,
            'os_licensing_cost': os_licensing_cost,
            'management_cost': management_cost,
            'total_monthly_cost': total_monthly_cost,
            'one_time_migration_cost': one_time_migration_cost,
            'estimated_monthly_savings': estimated_monthly_savings,
            'roi_months': roi_months
        }

def main():
    """Main application function"""
    render_enhanced_header()
    
    # Get configuration
    config = render_enhanced_sidebar_controls()
    
    # Initialize analyzer
    analyzer = EnhancedMigrationAnalyzer()
    
    # Run comprehensive analysis
    with st.spinner("🔬 Running comprehensive AWS migration analysis v2.0..."):
        analysis = analyzer.comprehensive_migration_analysis(config)
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 AWS Sizing", 
        "🤖 Agent Config",
        "🌐 Network Analysis",
        "💰 Cost Analysis",
        "📊 Performance Dashboard",
        "🖥️ On-Prem Performance",
        "☁️ AWS Performance"
    ])
    
    with tab1:
        render_enhanced_aws_recommendations(analysis, config)
    
    with tab2:
        render_agent_sizing_analysis(analysis, config)
    
    with tab3:
        render_enhanced_network_analysis(analysis, config)
    
    with tab4:
        render_enhanced_cost_analysis(analysis, config)
    
    with tab5:
        # Performance dashboard
        st.subheader("📊 Migration Performance Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Migration Throughput", 
                f"{analysis['migration_throughput_mbps']:.0f} Mbps",
                delta=f"{analysis['primary_tool'].upper()} optimized"
            )
        
        with col2:
            st.metric(
                "Estimated Time", 
                f"{analysis['estimated_migration_time_hours']:.1f} hours",
                delta=f"{config['database_size_gb']} GB database"
            )
        
        with col3:
            deployment = analysis['aws_sizing_recommendations']['deployment_recommendation']['recommendation']
            st.metric(
                "AWS Deployment", 
                deployment.upper(),
                delta=f"{analysis['aws_sizing_recommendations']['reader_writer_config']['total_instances']} instances"
            )
        
        with col4:
            st.metric(
                "Monthly Cost", 
                f"${analysis['cost_analysis']['total_monthly_cost']:.0f}",
                delta=f"${analysis['cost_analysis']['estimated_monthly_savings']:.0f} potential savings"
            )
        
        # Performance comparison chart
        st.markdown("**📊 Throughput Analysis:**")
        
        throughput_data = {
            'Component': ['Hardware NIC', 'Network Path', 'Migration Agent', 'Effective Migration'],
            'Throughput (Mbps)': [
                analysis['hardware_performance']['os_adjusted_throughput'],
                analysis['network_performance']['effective_bandwidth_mbps'],
                analysis['agent_analysis']['effective_throughput'],
                analysis['migration_throughput_mbps']
            ]
        }
        
        fig = px.bar(throughput_data, x='Component', y='Throughput (Mbps)',
                    title="Migration Throughput Bottleneck Analysis",
                    color='Throughput (Mbps)',
                    color_continuous_scale='RdYlGn')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional performance insights
        st.markdown("**🔍 Performance Insights:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="performance-card">
                <h4>🚀 Migration Performance Summary</h4>
                <p><strong>Bottleneck:</strong> {analysis['agent_analysis']['bottleneck'].title()}</p>
                <p><strong>Agent Utilization:</strong> {analysis['agent_analysis']['throughput_impact']*100:.1f}%</p>
                <p><strong>Network Quality:</strong> {analysis['network_performance']['network_quality_score']:.1f}/100</p>
                <p><strong>OS Efficiency:</strong> {analysis['hardware_performance']['os_efficiency_factor']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # On-prem vs AWS performance preview
            onprem_score = analysis['onprem_performance']['performance_score']
            aws_score = analysis['aws_performance_analysis']['performance_comparison']['overall_score_comparison']['aws_score']
            
            st.markdown(f"""
            <div class="performance-card">
                <h4>📈 Performance Comparison Preview</h4>
                <p><strong>On-Premises Score:</strong> {onprem_score:.1f}/100</p>
                <p><strong>Expected AWS Score:</strong> {aws_score:.1f}/100</p>
                <p><strong>Performance Change:</strong> {aws_score - onprem_score:+.1f} points</p>
                <p><strong>Status:</strong> {'📈 Improvement' if aws_score > onprem_score else '📉 Decline' if aws_score < onprem_score else '➡️ Similar'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab6:
        render_onprem_performance_tab(analysis['onprem_performance'], config)
    
    with tab7:
        render_aws_performance_tab(analysis['aws_performance_analysis'], config)
    
    # Summary footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        ☁️ <strong>Migration Summary:</strong> {analysis['migration_type'].title()} migration using AWS {analysis['primary_tool'].upper()} 
        • 🕒 <strong>Time:</strong> {analysis['estimated_migration_time_hours']:.1f} hours 
        • 💰 <strong>Monthly Cost:</strong> ${analysis['cost_analysis']['total_monthly_cost']:.0f} 
        • 🎯 <strong>AWS:</strong> {analysis['aws_sizing_recommendations']['deployment_recommendation']['recommendation'].upper()}
        • 📊 <strong>Performance:</strong> {analysis['onprem_performance']['performance_score']:.1f} → {analysis['aws_performance_analysis']['performance_comparison']['overall_score_comparison']['aws_score']:.1f}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()