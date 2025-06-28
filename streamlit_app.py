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
import os
import platform
import subprocess

# Page configuration
st.set_page_config(
    page_title="AWS Enterprise Database Migration Analyzer",
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
    
    .migration-tool-card {
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
            'security_overhead': os_config['security_overhead']
        }

class AWSMigrationManager:
    """Manage AWS-specific migration strategies and tools"""
    
    def __init__(self):
        self.migration_types = {
            'homogeneous': {
                'name': 'Homogeneous Migration',
                'description': 'Same database engine (e.g., MySQL to RDS MySQL)',
                'complexity_factor': 0.3,
                'time_factor': 0.7,
                'risk_factor': 0.2,
                'tools': ['aws_dms', 'native_replication'],
                'schema_conversion_required': False,
                'application_changes_required': False
            },
            'heterogeneous': {
                'name': 'Heterogeneous Migration',
                'description': 'Different database engines (e.g., Oracle to PostgreSQL)',
                'complexity_factor': 0.8,
                'time_factor': 1.4,
                'risk_factor': 0.7,
                'tools': ['aws_dms', 'aws_sct'],
                'schema_conversion_required': True,
                'application_changes_required': True
            }
        }
        
        self.aws_migration_tools = {
            'aws_dms': {
                'name': 'AWS Database Migration Service',
                'best_for': ['continuous_replication', 'minimal_downtime', 'heterogeneous'],
                'throughput_efficiency': 0.85,
                'cpu_overhead': 0.15,
                'memory_overhead': 0.10,
                'network_efficiency': 0.90,
                'setup_complexity': 0.4,
                'ongoing_cost_factor': 1.2,
                'max_concurrent_tasks': 200,
                'supported_engines': ['mysql', 'postgresql', 'oracle', 'sqlserver', 'mongodb']
            },
            'aws_datasync': {
                'name': 'AWS DataSync',
                'best_for': ['bulk_transfer', 'file_based', 'initial_migration'],
                'throughput_efficiency': 0.95,
                'cpu_overhead': 0.05,
                'memory_overhead': 0.03,
                'network_efficiency': 0.98,
                'setup_complexity': 0.2,
                'ongoing_cost_factor': 0.8,
                'max_concurrent_tasks': 50,
                'supported_engines': ['mysql', 'postgresql', 'mongodb']  # For file-based migrations
            },
            'native_replication': {
                'name': 'Native Database Replication',
                'best_for': ['homogeneous', 'high_performance', 'custom_control'],
                'throughput_efficiency': 0.98,
                'cpu_overhead': 0.08,
                'memory_overhead': 0.05,
                'network_efficiency': 0.95,
                'setup_complexity': 0.7,
                'ongoing_cost_factor': 1.0,
                'max_concurrent_tasks': 1000,
                'supported_engines': ['mysql', 'postgresql', 'oracle', 'sqlserver']
            }
        }
        
        self.aws_deployment_options = {
            'rds': {
                'name': 'Amazon RDS',
                'management_overhead': 0.1,
                'performance_factor': 0.92,
                'cost_factor': 1.3,
                'scalability_factor': 0.85,
                'backup_automation': True,
                'monitoring_included': True,
                'patch_management': True,
                'multi_az_support': True,
                'read_replica_support': True
            },
            'ec2': {
                'name': 'Amazon EC2',
                'management_overhead': 0.4,
                'performance_factor': 0.98,
                'cost_factor': 1.0,
                'scalability_factor': 0.95,
                'backup_automation': False,
                'monitoring_included': False,
                'patch_management': False,
                'multi_az_support': False,
                'read_replica_support': False
            }
        }
    
    def recommend_migration_approach(self, source_engine: str, target_engine: str, 
                                   database_size_gb: int, downtime_tolerance_minutes: int,
                                   performance_requirements: str) -> Dict:
        """Recommend optimal migration approach based on requirements"""
        
        # Determine migration type
        migration_type = 'homogeneous' if source_engine == target_engine else 'heterogeneous'
        migration_config = self.migration_types[migration_type]
        
        # Recommend tools based on requirements
        recommended_tools = []
        
        if migration_type == 'homogeneous':
            if downtime_tolerance_minutes < 30 and database_size_gb > 1000:
                recommended_tools = ['native_replication', 'aws_dms']
            elif database_size_gb < 100:
                recommended_tools = ['aws_datasync', 'aws_dms']
            else:
                recommended_tools = ['aws_dms', 'native_replication']
        else:  # heterogeneous
            recommended_tools = ['aws_dms']  # Primary choice for heterogeneous
        
        # Calculate complexity and time estimates
        base_time_hours = (database_size_gb / 100) * migration_config['time_factor']
        complexity_score = migration_config['complexity_factor'] * 100
        
        # Recommend AWS deployment
        aws_deployment = self._recommend_aws_deployment(
            target_engine, database_size_gb, performance_requirements
        )
        
        return {
            'migration_type': migration_type,
            'migration_config': migration_config,
            'recommended_tools': recommended_tools,
            'primary_tool': recommended_tools[0] if recommended_tools else 'aws_dms',
            'estimated_time_hours': base_time_hours,
            'complexity_score': complexity_score,
            'aws_deployment': aws_deployment
        }
    
    def _recommend_aws_deployment(self, database_engine: str, database_size_gb: int, 
                                performance_requirements: str) -> Dict:
        """Recommend RDS vs EC2 deployment"""
        
        rds_score = 0
        ec2_score = 0
        
        # Size considerations
        if database_size_gb < 1000:
            rds_score += 30
        elif database_size_gb > 5000:
            ec2_score += 20
        
        # Performance requirements
        if performance_requirements == 'high':
            ec2_score += 25
            rds_score += 10
        elif performance_requirements == 'standard':
            rds_score += 25
        
        # Database engine considerations
        if database_engine in ['mysql', 'postgresql']:
            rds_score += 20
        elif database_engine == 'oracle':
            ec2_score += 30  # Oracle licensing complexity
        
        # Management preferences (assume prefer managed service)
        rds_score += 30
        
        recommendation = 'rds' if rds_score > ec2_score else 'ec2'
        confidence = abs(rds_score - ec2_score) / max(rds_score, ec2_score, 1)
        
        # Recommend read replicas
        read_replicas = self._recommend_read_replicas(database_size_gb, performance_requirements)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rds_score': rds_score,
            'ec2_score': ec2_score,
            'read_replicas': read_replicas,
            'reasoning': self._generate_deployment_reasoning(recommendation, rds_score, ec2_score)
        }
    
    def _recommend_read_replicas(self, database_size_gb: int, performance_requirements: str) -> Dict:
        """Recommend number of read replicas and writers"""
        
        # Base recommendations
        writers = 1  # Start with single writer
        readers = 0
        
        # Size-based scaling
        if database_size_gb > 1000:
            readers += 1
        if database_size_gb > 5000:
            readers += 1
        if database_size_gb > 10000:
            readers += 2
            writers = 2  # Consider multi-writer for very large DBs
        
        # Performance-based scaling
        if performance_requirements == 'high':
            readers += 2
        elif performance_requirements == 'standard':
            readers += 1
        
        # Ensure minimum recommendations
        readers = max(1, readers)
        
        return {
            'writers': writers,
            'readers': readers,
            'total_instances': writers + readers,
            'reasoning': f"Based on {database_size_gb}GB size and {performance_requirements} performance requirements"
        }
    
    def _generate_deployment_reasoning(self, recommendation: str, rds_score: int, ec2_score: int) -> str:
        """Generate human-readable reasoning for deployment recommendation"""
        
        if recommendation == 'rds':
            return f"RDS recommended (score: {rds_score} vs {ec2_score}) for managed service benefits, automated backups, and easier scaling"
        else:
            return f"EC2 recommended (score: {ec2_score} vs {rds_score}) for maximum performance control, custom configurations, and potential cost savings"

class NetworkPathManager:
    """Manage specific network paths for AWS migration"""
    
    def __init__(self):
        self.network_paths = {
            'nonprod_sj_to_usw2': {
                'name': 'Non-Prod: San Jose to US-West-2',
                'source': 'San Jose',
                'destination': 'AWS US-West-2',
                'environment': 'non-production',
                'segments': [
                    {
                        'name': 'San Jose to AWS US-West-2',
                        'bandwidth_mbps': 2000,  # 2 Gbps
                        'latency_ms': 25,
                        'reliability': 0.99,
                        'connection_type': 'internet',
                        'cost_factor': 1.0
                    }
                ]
            },
            'prod_sa_to_usw2': {
                'name': 'Prod: San Antonio via San Jose to US-West-2',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2',
                'environment': 'production',
                'segments': [
                    {
                        'name': 'San Antonio to San Jose',
                        'bandwidth_mbps': 10000,  # 10 Gbps
                        'latency_ms': 15,
                        'reliability': 0.995,
                        'connection_type': 'private_line',
                        'cost_factor': 2.0
                    },
                    {
                        'name': 'San Jose to AWS US-West-2 (DX)',
                        'bandwidth_mbps': 10000,  # 10 Gbps DX
                        'latency_ms': 12,
                        'reliability': 0.999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 3.0
                    }
                ]
            }
        }
    
    def calculate_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """Calculate network path performance characteristics"""
        
        path = self.network_paths[path_key]
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        # Calculate end-to-end performance
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        
        for segment in path['segments']:
            # Base metrics
            total_latency += segment['latency_ms']
            min_bandwidth = min(min_bandwidth, segment['bandwidth_mbps'])
            total_reliability *= segment['reliability']
            total_cost_factor += segment['cost_factor']
            
            # Time-of-day adjustments
            if segment['connection_type'] == 'internet':
                # Internet connections more affected by business hours
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.3
                else:
                    congestion_factor = 0.8
                segment_bandwidth = segment['bandwidth_mbps'] / congestion_factor
                segment_latency = segment['latency_ms'] * congestion_factor
            elif segment['connection_type'] == 'direct_connect':
                # DX connections very stable
                segment_bandwidth = segment['bandwidth_mbps'] * 0.98
                segment_latency = segment['latency_ms'] * 1.02
            else:  # private_line
                # Private lines moderately affected
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.1
                else:
                    congestion_factor = 0.95
                segment_bandwidth = segment['bandwidth_mbps'] / congestion_factor
                segment_latency = segment['latency_ms'] * congestion_factor
        
        # Calculate effective throughput
        effective_bandwidth = min_bandwidth * total_reliability
        
        # Network quality score
        latency_score = max(0, 100 - total_latency)
        bandwidth_score = min(100, (effective_bandwidth / 1000) * 10)  # Score based on Gbps
        reliability_score = total_reliability * 100
        
        network_quality = (latency_score * 0.3 + bandwidth_score * 0.4 + reliability_score * 0.3)
        
        return {
            'path_name': path['name'],
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': effective_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': network_quality,
            'total_cost_factor': total_cost_factor,
            'segments': path['segments'],
            'environment': path['environment']
        }

class EnhancedFlexibleHardwareManager:
    """Enhanced hardware manager with OS support"""
    
    def __init__(self):
        # Inherit from original but extend with OS awareness
        self.nic_types = {
            'gigabit_copper': {
                'name': 'Gigabit Ethernet (Copper)',
                'max_speed': 1000,
                'latency_factor': 1.0,
                'cpu_overhead': 0.08,
                'reliability': 0.98,
                'cost_factor': 1.0
            },
            'gigabit_fiber': {
                'name': 'Gigabit Ethernet (Fiber)',
                'max_speed': 1000,
                'latency_factor': 0.95,
                'cpu_overhead': 0.06,
                'reliability': 0.99,
                'cost_factor': 1.5
            },
            '10g_copper': {
                'name': '10Gb Ethernet (Copper)',
                'max_speed': 10000,
                'latency_factor': 0.8,
                'cpu_overhead': 0.12,
                'reliability': 0.985,
                'cost_factor': 3.0
            },
            '10g_fiber': {
                'name': '10Gb Ethernet (Fiber)',
                'max_speed': 10000,
                'latency_factor': 0.75,
                'cpu_overhead': 0.10,
                'reliability': 0.995,
                'cost_factor': 4.0
            },
            '25g_fiber': {
                'name': '25Gb Ethernet (Fiber)',
                'max_speed': 25000,
                'latency_factor': 0.7,
                'cpu_overhead': 0.15,
                'reliability': 0.997,
                'cost_factor': 8.0
            },
            '40g_fiber': {
                'name': '40Gb Ethernet (Fiber)',
                'max_speed': 40000,
                'latency_factor': 0.65,
                'cpu_overhead': 0.18,
                'reliability': 0.998,
                'cost_factor': 12.0
            }
        }
        
        self.server_platforms = {
            'physical': {
                'name': 'Physical Server',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.95,
                'io_efficiency': 1.0,
                'network_efficiency': 0.98,
                'base_overhead': 0.02
            },
            'vmware': {
                'name': 'VMware Virtual Machine',
                'cpu_efficiency': 0.82,
                'memory_efficiency': 0.78,
                'io_efficiency': 0.75,
                'network_efficiency': 0.80,
                'base_overhead': 0.18
            }
        }
    
    def calculate_hardware_performance_with_os(self, config: Dict, os_manager: OSPerformanceManager) -> Dict:
        """Calculate performance with OS considerations"""
        
        # Get base hardware performance
        base_perf = self.calculate_hardware_performance(config)
        
        # Get OS-specific impact
        os_impact = os_manager.calculate_os_performance_impact(
            config['operating_system'], 
            config['server_type'], 
            config['database_engine']
        )
        
        # Apply OS efficiency to base performance
        os_adjusted_throughput = base_perf['actual_throughput'] * os_impact['total_efficiency']
        
        return {
            **base_perf,
            'os_impact': os_impact,
            'os_adjusted_throughput': os_adjusted_throughput,
            'os_efficiency_factor': os_impact['total_efficiency'],
            'os_licensing_cost': os_impact['licensing_cost_factor'],
            'os_management_complexity': os_impact['management_complexity']
        }
    
    def calculate_hardware_performance(self, config: Dict) -> Dict:
        """Original hardware performance calculation"""
        
        platform = self.server_platforms[config['server_type']]
        nic = self.nic_types[config['nic_type']]
        
        # RAM Performance Impact
        ram_gb = config['ram_gb']
        ram_performance = min(1.0, ram_gb / 64)
        
        # CPU Performance Impact
        cpu_cores = config['cpu_cores']
        cpu_ghz = config['cpu_ghz']
        cpu_performance = min(1.0, (cpu_cores * cpu_ghz) / 32)
        
        # NIC Performance Impact
        nic_performance = min(1.0, config['nic_speed'] / nic['max_speed'])
        
        # Platform efficiency
        platform_efficiency = (
            platform['cpu_efficiency'] * platform['memory_efficiency'] * 
            platform['io_efficiency'] * platform['network_efficiency']
        )
        
        # Combined performance score
        overall_performance = (
            ram_performance * 0.25 +
            cpu_performance * 0.30 +
            nic_performance * 0.25 +
            platform_efficiency * 0.20
        )
        
        # Calculate actual throughput
        theoretical_throughput = config['nic_speed'] * nic_performance
        actual_throughput = theoretical_throughput * overall_performance * (1 - nic['cpu_overhead'])
        
        # VMware specific calculations
        vmware_impact = 0
        if config['server_type'] == 'vmware':
            vmware_impact = self._calculate_detailed_vmware_impact(config)
            actual_throughput *= (1 - vmware_impact)
        
        return {
            'ram_performance': ram_performance,
            'cpu_performance': cpu_performance,
            'nic_performance': nic_performance,
            'platform_efficiency': platform_efficiency,
            'overall_performance': overall_performance,
            'theoretical_throughput': theoretical_throughput,
            'actual_throughput': actual_throughput,
            'vmware_impact': vmware_impact,
            'nic_characteristics': nic,
            'platform_characteristics': platform
        }
    
    def _calculate_detailed_vmware_impact(self, config: Dict) -> float:
        """Calculate detailed VMware impact"""
        
        base_impact = 0.15
        
        if config['ram_gb'] < 32:
            base_impact += 0.08
        elif config['ram_gb'] < 16:
            base_impact += 0.15
        
        if config['cpu_cores'] < 4:
            base_impact += 0.10
        elif config['cpu_cores'] < 8:
            base_impact += 0.05
        
        if config['nic_speed'] < 1000:
            base_impact += 0.12
        elif config['nic_speed'] < 10000:
            base_impact += 0.06
        
        return min(0.4, base_impact)

class EnhancedMigrationAnalyzer:
    """Enhanced migration analyzer with AWS and OS support"""
    
    def __init__(self, anthropic_api_key: str = None):
        self.os_manager = OSPerformanceManager()
        self.aws_manager = AWSMigrationManager()
        self.network_manager = NetworkPathManager()
        self.hardware_manager = EnhancedFlexibleHardwareManager()
        
        # Database engines with enhanced characteristics
        self.database_engines = {
            'mysql': {
                'name': 'MySQL',
                'memory_efficiency': 0.85,
                'cpu_efficiency': 0.88,
                'network_sensitivity': 0.7,
                'connection_overhead_kb': 4,
                'optimal_ram_per_gb_db': 0.1,
                'cpu_intensive': False,
                'supports_compression': True,
                'aws_rds_support': True,
                'migration_complexity': 0.3
            },
            'postgresql': {
                'name': 'PostgreSQL',
                'memory_efficiency': 0.90,
                'cpu_efficiency': 0.92,
                'network_sensitivity': 0.75,
                'connection_overhead_kb': 8,
                'optimal_ram_per_gb_db': 0.15,
                'cpu_intensive': True,
                'supports_compression': True,
                'aws_rds_support': True,
                'migration_complexity': 0.4
            },
            'oracle': {
                'name': 'Oracle Database',
                'memory_efficiency': 0.95,
                'cpu_efficiency': 0.94,
                'network_sensitivity': 0.6,
                'connection_overhead_kb': 12,
                'optimal_ram_per_gb_db': 0.2,
                'cpu_intensive': True,
                'supports_compression': True,
                'aws_rds_support': True,
                'migration_complexity': 0.7
            },
            'sqlserver': {
                'name': 'SQL Server',
                'memory_efficiency': 0.88,
                'cpu_efficiency': 0.90,
                'network_sensitivity': 0.65,
                'connection_overhead_kb': 6,
                'optimal_ram_per_gb_db': 0.12,
                'cpu_intensive': False,
                'supports_compression': True,
                'aws_rds_support': True,
                'migration_complexity': 0.5
            },
            'mongodb': {
                'name': 'MongoDB',
                'memory_efficiency': 0.82,
                'cpu_efficiency': 0.85,
                'network_sensitivity': 0.8,
                'connection_overhead_kb': 3,
                'optimal_ram_per_gb_db': 0.08,
                'cpu_intensive': False,
                'supports_compression': True,
                'aws_rds_support': False,  # DocumentDB instead
                'migration_complexity': 0.6
            }
        }
    
    def comprehensive_migration_analysis(self, config: Dict) -> Dict:
        """Comprehensive migration analysis with all enhancements"""
        
        # Hardware and OS performance analysis
        hardware_perf = self.hardware_manager.calculate_hardware_performance_with_os(
            config, self.os_manager
        )
        
        # Network path analysis
        network_path = config.get('network_path', 'nonprod_sj_to_usw2')
        network_perf = self.network_manager.calculate_path_performance(network_path)
        
        # AWS migration strategy
        migration_strategy = self.aws_manager.recommend_migration_approach(
            config.get('source_database_engine', config['database_engine']),
            config['database_engine'],
            config['database_size_gb'],
            config.get('downtime_tolerance_minutes', 60),
            config.get('performance_requirements', 'standard')
        )
        
        # Calculate effective migration throughput
        base_throughput = hardware_perf['os_adjusted_throughput']
        network_throughput = min(base_throughput, network_perf['effective_bandwidth_mbps'])
        
        # Apply migration tool efficiency
        tool_config = self.aws_manager.aws_migration_tools[migration_strategy['primary_tool']]
        migration_throughput = network_throughput * tool_config['throughput_efficiency']
        
        # Calculate migration time
        database_size_gb = config['database_size_gb']
        migration_time_hours = (database_size_gb * 8 * 1000) / (migration_throughput * 3600)
        
        # OS comparison analysis
        os_comparison = self._compare_operating_systems(config)
        
        # Platform comparison (Physical vs VMware with OS)
        platform_comparison = self._compare_platforms_with_os(config)
        
        return {
            'hardware_performance': hardware_perf,
            'network_performance': network_perf,
            'migration_strategy': migration_strategy,
            'migration_throughput_mbps': migration_throughput,
            'estimated_migration_time_hours': migration_time_hours,
            'os_comparison': os_comparison,
            'platform_comparison': platform_comparison,
            'aws_recommendations': self._generate_detailed_aws_recommendations(config, migration_strategy),
            'cost_analysis': self._calculate_migration_costs(config, migration_strategy, hardware_perf)
        }
    
    def _compare_operating_systems(self, config: Dict) -> Dict:
        """Compare different operating systems for the same configuration"""
        
        os_options = list(self.os_manager.operating_systems.keys())
        comparison = {}
        
        for os_type in os_options:
            test_config = config.copy()
            test_config['operating_system'] = os_type
            
            perf = self.hardware_manager.calculate_hardware_performance_with_os(
                test_config, self.os_manager
            )
            
            comparison[os_type] = {
                'name': self.os_manager.operating_systems[os_type]['name'],
                'throughput_mbps': perf['os_adjusted_throughput'],
                'efficiency': perf['os_efficiency_factor'],
                'licensing_cost_factor': perf['os_licensing_cost'],
                'management_complexity': perf['os_management_complexity']
            }
        
        # Find best performing OS
        best_os = max(comparison.keys(), key=lambda x: comparison[x]['throughput_mbps'])
        
        return {
            'comparison': comparison,
            'best_performing_os': best_os,
            'current_os_rank': sorted(os_options, key=lambda x: comparison[x]['throughput_mbps'], reverse=True).index(config['operating_system']) + 1
        }
    
    def _compare_platforms_with_os(self, config: Dict) -> Dict:
        """Compare Physical vs VMware with OS considerations"""
        
        platforms = ['physical', 'vmware']
        comparison = {}
        
        for platform in platforms:
            test_config = config.copy()
            test_config['server_type'] = platform
            
            perf = self.hardware_manager.calculate_hardware_performance_with_os(
                test_config, self.os_manager
            )
            
            comparison[platform] = {
                'throughput_mbps': perf['os_adjusted_throughput'],
                'efficiency': perf['os_efficiency_factor'],
                'platform_efficiency': perf['platform_efficiency'],
                'vmware_overhead': perf.get('vmware_impact', 0) * 100
            }
        
        # Calculate performance gap
        performance_gap = ((comparison['physical']['throughput_mbps'] - 
                          comparison['vmware']['throughput_mbps']) / 
                         comparison['physical']['throughput_mbps']) * 100
        
        return {
            'comparison': comparison,
            'performance_gap_percent': performance_gap,
            'recommendation': 'physical' if performance_gap > 20 else 'vmware_acceptable'
        }
    
    def _generate_detailed_aws_recommendations(self, config: Dict, migration_strategy: Dict) -> Dict:
        """Generate detailed AWS recommendations"""
        
        deployment = migration_strategy['aws_deployment']
        
        # Instance type recommendations
        if deployment['recommendation'] == 'rds':
            instance_recommendations = self._recommend_rds_instances(config)
        else:
            instance_recommendations = self._recommend_ec2_instances(config)
        
        # Region and AZ recommendations
        region_recommendations = {
            'primary_region': 'us-west-2',
            'backup_region': 'us-west-1',
            'multi_az': deployment['recommendation'] == 'rds',
            'availability_zones': ['us-west-2a', 'us-west-2b', 'us-west-2c']
        }
        
        return {
            'deployment_type': deployment['recommendation'],
            'instance_recommendations': instance_recommendations,
            'region_recommendations': region_recommendations,
            'read_replica_strategy': deployment['read_replicas'],
            'backup_strategy': self._recommend_backup_strategy(config),
            'security_recommendations': self._recommend_security_configuration(config)
        }
    
    def _recommend_rds_instances(self, config: Dict) -> Dict:
        """Recommend RDS instance types"""
        
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        
        # Base instance selection logic
        if database_size_gb < 500:
            if performance_req == 'high':
                instance_class = 'db.r6g.xlarge'
            else:
                instance_class = 'db.r6g.large'
        elif database_size_gb < 2000:
            if performance_req == 'high':
                instance_class = 'db.r6g.2xlarge'
            else:
                instance_class = 'db.r6g.xlarge'
        else:
            if performance_req == 'high':
                instance_class = 'db.r6g.4xlarge'
            else:
                instance_class = 'db.r6g.2xlarge'
        
        return {
            'primary_instance': instance_class,
            'read_replica_instance': instance_class,  # Same for consistency
            'storage_type': 'gp3' if database_size_gb < 1000 else 'io1',
            'storage_size_gb': max(database_size_gb * 1.2, 100),  # 20% overhead
            'backup_retention_days': 7,
            'multi_az': True
        }
    
    def _recommend_ec2_instances(self, config: Dict) -> Dict:
        """Recommend EC2 instance types"""
        
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        
        # Base instance selection logic
        if database_size_gb < 500:
            if performance_req == 'high':
                instance_type = 'r6i.xlarge'
            else:
                instance_type = 'r6i.large'
        elif database_size_gb < 2000:
            if performance_req == 'high':
                instance_type = 'r6i.2xlarge'
            else:
                instance_type = 'r6i.xlarge'
        else:
            if performance_req == 'high':
                instance_type = 'r6i.4xlarge'
            else:
                instance_type = 'r6i.2xlarge'
        
        return {
            'primary_instance': instance_type,
            'read_replica_instance': instance_type,
            'storage_type': 'gp3',
            'storage_size_gb': max(database_size_gb * 1.5, 100),  # 50% overhead for OS and logs
            'ebs_optimized': True,
            'enhanced_networking': True
        }
    
    def _recommend_backup_strategy(self, config: Dict) -> Dict:
        """Recommend backup strategy"""
        
        return {
            'automated_backup': True,
            'backup_retention_days': 30 if config.get('environment') == 'production' else 7,
            'point_in_time_recovery': True,
            'cross_region_backup': config.get('environment') == 'production',
            'backup_window': '03:00-04:00',  # Off-peak hours
            'maintenance_window': 'Sun:04:00-Sun:05:00'
        }
    
    def _recommend_security_configuration(self, config: Dict) -> Dict:
        """Recommend security configuration"""
        
        return {
            'encryption_at_rest': True,
            'encryption_in_transit': True,
            'vpc_security_groups': True,
            'subnet_groups': 'private',
            'iam_database_authentication': True,
            'performance_insights': True,
            'enhanced_monitoring': True,
            'deletion_protection': config.get('environment') == 'production'
        }
    
    def _calculate_migration_costs(self, config: Dict, migration_strategy: Dict, hardware_perf: Dict) -> Dict:
        """Calculate comprehensive migration costs"""
        
        # Base AWS costs (simplified)
        database_size_gb = config['database_size_gb']
        
        # RDS vs EC2 costs
        if migration_strategy['aws_deployment']['recommendation'] == 'rds':
            monthly_compute_cost = database_size_gb * 0.5  # Simplified
            managed_service_premium = monthly_compute_cost * 0.3
        else:
            monthly_compute_cost = database_size_gb * 0.3
            managed_service_premium = 0
        
        # Storage costs
        storage_cost = database_size_gb * 0.1
        
        # Network costs
        network_path = config.get('network_path', 'nonprod_sj_to_usw2')
        if 'prod' in network_path:
            network_cost = 500  # DX connection cost
        else:
            network_cost = 100  # Internet transfer cost
        
        # Migration tool costs
        tool_cost = migration_strategy['complexity_score'] * 10
        
        # OS licensing costs
        os_licensing_cost = hardware_perf['os_licensing_cost'] * 200  # Monthly
        
        total_monthly_cost = (monthly_compute_cost + managed_service_premium + 
                            storage_cost + network_cost + os_licensing_cost)
        
        return {
            'monthly_compute_cost': monthly_compute_cost,
            'managed_service_premium': managed_service_premium,
            'storage_cost': storage_cost,
            'network_cost': network_cost,
            'migration_tool_cost': tool_cost,
            'os_licensing_cost': os_licensing_cost,
            'total_monthly_cost': total_monthly_cost,
            'annual_cost': total_monthly_cost * 12
        }

def render_enhanced_header():
    """Render enhanced header for AWS migration analyzer"""
    st.markdown("""
    <div class="main-header">
        <h1>☁️ AWS Enterprise Database Migration Analyzer</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">Windows vs Linux • Physical vs VMware • Homogeneous vs Heterogeneous • AWS DMS vs DataSync</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Real-time Network Paths • RDS vs EC2 Recommendations • Cost Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar_controls():
    """Render enhanced sidebar with all new options"""
    st.sidebar.header("☁️ AWS Migration Configuration")
    
    # Operating System Selection
    st.sidebar.subheader("💻 Operating System")
    operating_system = st.sidebar.selectbox(
        "OS Selection",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3,  # Default to RHEL 9
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
    
    # Hardware Configuration (simplified for space)
    st.sidebar.subheader("⚙️ Hardware")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256], index=2)
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32], index=2)
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
    # Network Interface
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber"],
        index=3
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000, '25g_fiber': 25000
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
    
    database_size_gb = st.sidebar.number_input("Database Size (GB)", min_value=100, max_value=50000, value=1000, step=100)
    
    # Migration Parameters
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", min_value=1, max_value=480, value=60)
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"])
    
    # Network Path Selection
    st.sidebar.subheader("🌐 Network Path")
    network_path = st.sidebar.selectbox(
        "Migration Path",
        ["nonprod_sj_to_usw2", "prod_sa_to_usw2"],
        format_func=lambda x: {
            'nonprod_sj_to_usw2': 'Non-Prod: San Jose → AWS US-West-2 (2Gbps)',
            'prod_sa_to_usw2': 'Prod: San Antonio → San Jose → AWS US-West-2 (10Gbps DX)'
        }[x]
    )
    
    # Environment
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"])
    
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
        'environment': environment
    }

def render_os_platform_comparison(analysis: Dict, config: Dict):
    """Render OS and platform comparison analysis"""
    st.subheader("💻 Operating System & Platform Impact Analysis")
    
    # Current configuration summary
    col1, col2, col3, col4 = st.columns(4)
    
    current_os_name = analysis['os_comparison']['comparison'][config['operating_system']]['name']
    
    with col1:
        st.markdown(f"""
        <div class="os-comparison-card">
            <h4>Current Configuration</h4>
            <p><strong>OS:</strong> {current_os_name}</p>
            <p><strong>Platform:</strong> {config['server_type'].title()}</p>
            <p><strong>Throughput:</strong> {analysis['hardware_performance']['os_adjusted_throughput']:.0f} Mbps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_os = analysis['os_comparison']['best_performing_os']
        best_os_name = analysis['os_comparison']['comparison'][best_os]['name']
        best_throughput = analysis['os_comparison']['comparison'][best_os]['throughput_mbps']
        
        st.markdown(f"""
        <div class="os-comparison-card">
            <h4>Best Performing OS</h4>
            <p><strong>OS:</strong> {best_os_name}</p>
            <p><strong>Throughput:</strong> {best_throughput:.0f} Mbps</p>
            <p><strong>Improvement:</strong> {((best_throughput / analysis['hardware_performance']['os_adjusted_throughput']) - 1) * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        platform_comp = analysis['platform_comparison']
        st.markdown(f"""
        <div class="os-comparison-card">
            <h4>Platform Comparison</h4>
            <p><strong>Physical:</strong> {platform_comp['comparison']['physical']['throughput_mbps']:.0f} Mbps</p>
            <p><strong>VMware:</strong> {platform_comp['comparison']['vmware']['throughput_mbps']:.0f} Mbps</p>
            <p><strong>Gap:</strong> {platform_comp['performance_gap_percent']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        os_rank = analysis['os_comparison']['current_os_rank']
        total_os = len(analysis['os_comparison']['comparison'])
        st.markdown(f"""
        <div class="os-comparison-card">
            <h4>Performance Ranking</h4>
            <p><strong>Current OS Rank:</strong> {os_rank} of {total_os}</p>
            <p><strong>Efficiency:</strong> {analysis['hardware_performance']['os_efficiency_factor']*100:.1f}%</p>
            <p><strong>License Cost:</strong> {analysis['hardware_performance']['os_licensing_cost']:.1f}x</p>
        </div>
        """, unsafe_allow_html=True)

def render_aws_migration_strategy(analysis: Dict, config: Dict):
    """Render AWS migration strategy and recommendations"""
    st.subheader("☁️ AWS Migration Strategy & Recommendations")
    
    migration_strategy = analysis['migration_strategy']
    aws_recommendations = analysis['aws_recommendations']
    
    # Migration type and tool recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        migration_type = migration_strategy['migration_type']
        migration_config = migration_strategy['migration_config']
        
        st.markdown(f"""
        <div class="aws-migration-card">
            <h4>🔄 Migration Analysis</h4>
            <p><strong>Type:</strong> {migration_config['name']}</p>
            <p><strong>Source:</strong> {config['source_database_engine'].upper()}</p>
            <p><strong>Target:</strong> {config['database_engine'].upper()}</p>
            <p><strong>Complexity:</strong> {migration_strategy['complexity_score']:.0f}/100</p>
            <p><strong>Est. Time:</strong> {analysis['estimated_migration_time_hours']:.1f} hours</p>
            <p><strong>Primary Tool:</strong> {migration_strategy['primary_tool'].replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        deployment = aws_recommendations['deployment_type']
        instance_rec = aws_recommendations['instance_recommendations']
        
        if deployment == 'rds':
            st.markdown(f"""
            <div class="rds-recommendation">
                <h4>🎯 AWS RDS Recommendation</h4>
                <p><strong>Instance:</strong> {instance_rec['primary_instance']}</p>
                <p><strong>Storage:</strong> {instance_rec['storage_type']} ({instance_rec['storage_size_gb']} GB)</p>
                <p><strong>Multi-AZ:</strong> {'Yes' if instance_rec['multi_az'] else 'No'}</p>
                <p><strong>Backup Retention:</strong> {instance_rec['backup_retention_days']} days</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ec2-recommendation">
                <h4>🎯 AWS EC2 Recommendation</h4>
                <p><strong>Instance:</strong> {instance_rec['primary_instance']}</p>
                <p><strong>Storage:</strong> {instance_rec['storage_type']} ({instance_rec['storage_size_gb']} GB)</p>
                <p><strong>EBS Optimized:</strong> {'Yes' if instance_rec['ebs_optimized'] else 'No'}</p>
                <p><strong>Enhanced Networking:</strong> {'Yes' if instance_rec['enhanced_networking'] else 'No'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Read replica recommendations
    read_replicas = aws_recommendations['read_replica_strategy']
    st.markdown(f"""
    <div class="aws-recommendation-card">
        <h4>📊 Read Replica Strategy</h4>
        <p><strong>Writers:</strong> {read_replicas['writers']} instance(s)</p>
        <p><strong>Readers:</strong> {read_replicas['readers']} instance(s)</p>
        <p><strong>Total Instances:</strong> {read_replicas['total_instances']}</p>
        <p><strong>Reasoning:</strong> {read_replicas['reasoning']}</p>
    </div>
    """, unsafe_allow_html=True)

def render_network_path_analysis(analysis: Dict, config: Dict):
    """Render network path analysis"""
    st.subheader("🌐 Network Path Performance Analysis")
    
    network_perf = analysis['network_performance']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="network-path-card">
            <h4>📍 Path Details</h4>
            <p><strong>Route:</strong> {network_perf['path_name']}</p>
            <p><strong>Environment:</strong> {network_perf['environment'].title()}</p>
            <p><strong>Segments:</strong> {len(network_perf['segments'])}</p>
            <p><strong>Total Latency:</strong> {network_perf['total_latency_ms']:.1f} ms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="network-path-card">
            <h4>📊 Performance Metrics</h4>
            <p><strong>Bandwidth:</strong> {network_perf['effective_bandwidth_mbps']:.0f} Mbps</p>
            <p><strong>Reliability:</strong> {network_perf['total_reliability']*100:.2f}%</p>
            <p><strong>Quality Score:</strong> {network_perf['network_quality_score']:.1f}/100</p>
            <p><strong>Cost Factor:</strong> {network_perf['total_cost_factor']:.1f}x</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        migration_throughput = analysis['migration_throughput_mbps']
        st.markdown(f"""
        <div class="network-path-card">
            <h4>🚀 Migration Impact</h4>
            <p><strong>Migration Throughput:</strong> {migration_throughput:.0f} Mbps</p>
            <p><strong>Estimated Time:</strong> {analysis['estimated_migration_time_hours']:.1f} hours</p>
            <p><strong>Network Utilization:</strong> {(migration_throughput / network_perf['effective_bandwidth_mbps']) * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Segment details
    st.markdown("**🔗 Network Segment Details:**")
    segment_data = []
    for i, segment in enumerate(network_perf['segments']):
        segment_data.append({
            'Segment': segment['name'],
            'Bandwidth (Mbps)': f"{segment['bandwidth_mbps']:,}",
            'Latency (ms)': f"{segment['latency_ms']:.1f}",
            'Type': segment['connection_type'].replace('_', ' ').title(),
            'Reliability': f"{segment['reliability']*100:.1f}%",
            'Cost Factor': f"{segment['cost_factor']:.1f}x"
        })
    
    st.dataframe(pd.DataFrame(segment_data), use_container_width=True, hide_index=True)

def render_migration_tools_comparison(analysis: Dict):
    """Render migration tools comparison"""
    st.subheader("🛠️ Migration Tools Comparison")
    
    # Create analyzer instance to access tool configurations
    analyzer = EnhancedMigrationAnalyzer()
    tools = analyzer.aws_manager.aws_migration_tools
    
    tool_comparison = []
    for tool_name, tool_config in tools.items():
        tool_comparison.append({
            'Tool': tool_config['name'],
            'Throughput Efficiency': f"{tool_config['throughput_efficiency']*100:.1f}%",
            'CPU Overhead': f"{tool_config['cpu_overhead']*100:.1f}%",
            'Network Efficiency': f"{tool_config['network_efficiency']*100:.1f}%",
            'Setup Complexity': f"{tool_config['setup_complexity']*100:.0f}/100",
            'Cost Factor': f"{tool_config['ongoing_cost_factor']:.1f}x",
            'Best For': ', '.join(tool_config['best_for'][:2])  # Show first 2 use cases
        })
    
    df_tools = pd.DataFrame(tool_comparison)
    st.dataframe(df_tools, use_container_width=True, hide_index=True)
    
    # Highlight recommended tool
    recommended_tool = analysis['migration_strategy']['primary_tool']
    st.info(f"**Recommended Tool:** {tools[recommended_tool]['name']} - {', '.join(tools[recommended_tool]['best_for'])}")

def render_cost_analysis(analysis: Dict, config: Dict):
    """Render comprehensive cost analysis"""
    st.subheader("💰 Migration Cost Analysis")
    
    cost_analysis = analysis['cost_analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Compute", f"${cost_analysis['monthly_compute_cost']:.0f}")
    
    with col2:
        st.metric("Storage Cost", f"${cost_analysis['storage_cost']:.0f}")
    
    with col3:
        st.metric("Network Cost", f"${cost_analysis['network_cost']:.0f}")
    
    with col4:
        st.metric("OS Licensing", f"${cost_analysis['os_licensing_cost']:.0f}")
    
    # Detailed cost breakdown
    st.markdown("**📊 Detailed Cost Breakdown:**")
    
    cost_breakdown = pd.DataFrame([
        {'Component': 'Compute', 'Monthly Cost': f"${cost_analysis['monthly_compute_cost']:.0f}", 'Annual Cost': f"${cost_analysis['monthly_compute_cost']*12:.0f}"},
        {'Component': 'Managed Service Premium', 'Monthly Cost': f"${cost_analysis['managed_service_premium']:.0f}", 'Annual Cost': f"${cost_analysis['managed_service_premium']*12:.0f}"},
        {'Component': 'Storage', 'Monthly Cost': f"${cost_analysis['storage_cost']:.0f}", 'Annual Cost': f"${cost_analysis['storage_cost']*12:.0f}"},
        {'Component': 'Network', 'Monthly Cost': f"${cost_analysis['network_cost']:.0f}", 'Annual Cost': f"${cost_analysis['network_cost']*12:.0f}"},
        {'Component': 'OS Licensing', 'Monthly Cost': f"${cost_analysis['os_licensing_cost']:.0f}", 'Annual Cost': f"${cost_analysis['os_licensing_cost']*12:.0f}"},
        {'Component': 'Migration Tools', 'Monthly Cost': f"${cost_analysis['migration_tool_cost']:.0f}", 'Annual Cost': 'One-time'},
        {'Component': 'TOTAL', 'Monthly Cost': f"${cost_analysis['total_monthly_cost']:.0f}", 'Annual Cost': f"${cost_analysis['annual_cost']:.0f}"}
    ])
    
    st.dataframe(cost_breakdown, use_container_width=True, hide_index=True)

def render_performance_charts(analysis: Dict, config: Dict):
    """Render performance visualization charts"""
    st.subheader("📊 Performance Impact Visualization")
    
    # OS Comparison Chart
    os_comparison = analysis['os_comparison']['comparison']
    os_names = [os_comparison[k]['name'] for k in os_comparison.keys()]
    os_throughputs = [os_comparison[k]['throughput_mbps'] for k in os_comparison.keys()]
    
    fig_os = go.Figure(data=[
        go.Bar(
            x=os_names,
            y=os_throughputs,
            marker_color=['#FF9900' if k == config['operating_system'] else '#4ECDC4' for k in os_comparison.keys()],
            text=[f"{tp:.0f} Mbps" for tp in os_throughputs],
            textposition='auto'
        )
    ])
    
    fig_os.update_layout(
        title="Operating System Performance Comparison",
        xaxis_title="Operating System",
        yaxis_title="Throughput (Mbps)",
        height=400
    )
    
    st.plotly_chart(fig_os, use_container_width=True)
    
    # Migration Time Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Platform comparison
        platform_comp = analysis['platform_comparison']['comparison']
        platforms = list(platform_comp.keys())
        platform_throughputs = [platform_comp[p]['throughput_mbps'] for p in platforms]
        
        fig_platform = go.Figure(data=[
            go.Bar(
                x=['Physical', 'VMware'],
                y=platform_throughputs,
                marker_color=['#232F3E', '#FF9900'],
                text=[f"{tp:.0f} Mbps" for tp in platform_throughputs],
                textposition='auto'
            )
        ])
        
        fig_platform.update_layout(
            title="Physical vs VMware Performance",
            yaxis_title="Throughput (Mbps)",
            height=350
        )
        
        st.plotly_chart(fig_platform, use_container_width=True)
    
    with col2:
        # Migration time by database size
        sizes = [500, 1000, 2000, 5000, 10000]
        times = []
        for size in sizes:
            # Estimate time based on current throughput
            time_hours = (size * 8 * 1000) / (analysis['migration_throughput_mbps'] * 3600)
            times.append(time_hours)
        
        fig_time = go.Figure(data=[
            go.Scatter(
                x=sizes,
                y=times,
                mode='lines+markers',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8)
            )
        ])
        
        fig_time.update_layout(
            title="Migration Time by Database Size",
            xaxis_title="Database Size (GB)",
            yaxis_title="Migration Time (Hours)",
            height=350
        )
        
        st.plotly_chart(fig_time, use_container_width=True)

def main():
    """Main application function"""
    render_enhanced_header()
    
    # Get configuration
    config = render_enhanced_sidebar_controls()
    
    # Initialize analyzer
    analyzer = EnhancedMigrationAnalyzer()
    
    # Run comprehensive analysis
    with st.spinner("🔬 Running comprehensive AWS migration analysis..."):
        analysis = analyzer.comprehensive_migration_analysis(config)
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔧 OS & Platform",
        "☁️ AWS Strategy", 
        "🌐 Network Path",
        "🛠️ Migration Tools",
        "💰 Cost Analysis",
        "📊 Performance Charts"
    ])
    
    with tab1:
        render_os_platform_comparison(analysis, config)
    
    with tab2:
        render_aws_migration_strategy(analysis, config)
    
    with tab3:
        render_network_path_analysis(analysis, config)
    
    with tab4:
        render_migration_tools_comparison(analysis)
    
    with tab5:
        render_cost_analysis(analysis, config)
    
    with tab6:
        render_performance_charts(analysis, config)
    
    # Summary metrics at the bottom
    st.markdown("---")
    st.subheader("📋 Migration Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Migration Throughput", 
            f"{analysis['migration_throughput_mbps']:.0f} Mbps",
            delta=f"vs {analysis['hardware_performance']['actual_throughput']:.0f} HW baseline"
        )
    
    with col2:
        st.metric(
            "Estimated Time", 
            f"{analysis['estimated_migration_time_hours']:.1f} hours",
            delta=f"{config['database_size_gb']} GB"
        )
    
    with col3:
        deployment = analysis['aws_recommendations']['deployment_type']
        st.metric(
            "AWS Deployment", 
            deployment.upper(),
            delta=f"{analysis['aws_recommendations']['read_replica_strategy']['total_instances']} instances"
        )
    
    with col4:
        st.metric(
            "Monthly Cost", 
            f"${analysis['cost_analysis']['total_monthly_cost']:.0f}",
            delta=f"${analysis['cost_analysis']['annual_cost']:.0f} annual"
        )
    
    with col5:
        os_rank = analysis['os_comparison']['current_os_rank']
        total_os = len(analysis['os_comparison']['comparison'])
        st.metric(
            "OS Performance Rank", 
            f"{os_rank} of {total_os}",
            delta=f"{analysis['hardware_performance']['os_efficiency_factor']*100:.1f}% efficiency"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        ☁️ AWS Enterprise Migration Analyzer • 💻 Windows/Linux Comparison • 🔄 Homogeneous/Heterogeneous • 🛠️ DMS/DataSync • 🌐 Real Network Paths
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()