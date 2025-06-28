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
            'security_overhead': os_config['security_overhead']
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
        
        # Cost analysis
        cost_analysis = self._calculate_comprehensive_costs(config, aws_sizing, agent_analysis, network_perf)
        
        return {
            'hardware_performance': hardware_perf,
            'network_performance': network_perf,
            'migration_type': migration_type,
            'primary_tool': primary_tool,
            'agent_analysis': agent_analysis,
            'migration_throughput_mbps': migration_throughput,
            'estimated_migration_time_hours': migration_time_hours,
            'aws_sizing_recommendations': aws_sizing,
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 AWS Sizing", 
        "🤖 Agent Config",
        "🌐 Network Analysis",
        "💰 Cost Analysis",
        "📊 Performance Dashboard"
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
    
    # Summary footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        ☁️ <strong>Migration Summary:</strong> {analysis['migration_type'].title()} migration using AWS {analysis['primary_tool'].upper()} 
        • 🕒 <strong>Time:</strong> {analysis['estimated_migration_time_hours']:.1f} hours 
        • 💰 <strong>Monthly Cost:</strong> ${analysis['cost_analysis']['total_monthly_cost']:.0f} 
        • 🎯 <strong>AWS:</strong> {analysis['aws_sizing_recommendations']['deployment_recommendation']['recommendation'].upper()}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()