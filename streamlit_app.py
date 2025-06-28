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
    page_title="AWS Enterprise Database Migration Analyzer AI v3.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling with corporate colors
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(30,58,138,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    .professional-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #fefdf8 0%, #fefce8 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f59e0b;
        border: 1px solid #e5e7eb;
    }
    
    .pricing-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #0ea5e9;
        border: 1px solid #e5e7eb;
    }
    
    .network-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #22c55e;
        border: 1px solid #e5e7eb;
    }
    
    .storage-card {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #a855f7;
        border: 1px solid #e5e7eb;
    }
    
    .performance-card {
        background: linear-gradient(135deg, #fef7f0 0%, #fed7aa 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f97316;
        border: 1px solid #e5e7eb;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #64748b;
        border: 1px solid #e5e7eb;
    }
    
    .api-status-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #374151;
        font-size: 0.9rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-card h4, .metric-card h5 {
        margin: 0 0 0.8rem 0;
        color: #1f2937;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0.4rem 0;
        color: #6b7280;
        line-height: 1.4;
    }
    
    .network-diagram-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .status-online { background-color: #22c55e; }
    .status-offline { background-color: #ef4444; }
    .status-warning { background-color: #f59e0b; }
    
    .enterprise-footer {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 6px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .analysis-section {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .analysis-section h4, .analysis-section h5 {
        margin: 0 0 1rem 0;
        color: #1f2937;
        font-weight: 600;
    }
    
    .analysis-section p {
        margin: 0.5rem 0;
        color: #6b7280;
        line-height: 1.5;
    }
    
    .analysis-section ul {
        margin: 0.8rem 0;
        padding-left: 1.5rem;
    }
    
    .analysis-section li {
        margin: 0.4rem 0;
        color: #6b7280;
        line-height: 1.4;
    }
    
    /* Compact metric cards */
    .compact-metric {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    .compact-metric h5 {
        margin: 0 0 0.5rem 0;
        color: #1f2937;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .compact-metric p {
        margin: 0.2rem 0;
        color: #6b7280;
        font-size: 0.85rem;
        line-height: 1.3;
    }
    
    /* PDF download section */
    .pdf-section {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 6px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        text-align: center;
    }
    
    /* Professional tables */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Form controls */
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    
    .stNumberInput > div > div {
        border-radius: 6px;
    }
    
    /* Professional buttons */
    .stButton > button {
        border-radius: 6px;
        border: none;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
    
    /* Professional tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .stSidebar .stSelectbox label,
    .stSidebar .stNumberInput label {
        font-weight: 500;
        color: #1f2937;
        font-size: 0.9rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .professional-card, .insight-card, .recommendation-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class APIStatus:
    anthropic_connected: bool = False
    aws_pricing_connected: bool = False
    aws_compute_optimizer_connected: bool = False
    last_update: datetime = None
    error_message: str = None

class AnthropicAIManager:
    """Enhanced Anthropic AI manager with improved error handling and connection"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or st.secrets.get("ANTHROPIC_API_KEY")
        self.client = None
        self.connected = False
        self.error_message = None
        
        if self.api_key:
            try:
                # Initialize the client with latest model
                self.client = anthropic.Anthropic(api_key=self.api_key)
                
                # Test the connection with a simple API call using current model
                test_message = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
                
                self.connected = True
                logger.info("Anthropic AI client initialized and tested successfully")
                
            except Exception as e:
                self.connected = False
                self.error_message = str(e)
                logger.error(f"Failed to initialize Anthropic client: {e}")
        else:
            self.connected = False
            self.error_message = "No API key provided"
            logger.warning("No Anthropic API key found in secrets")
    
    async def analyze_migration_workload(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced AI-powered workload analysis with detailed insights"""
        if not self.connected:
            return self._fallback_workload_analysis(config, performance_data)
        
        try:
            # Enhanced prompt for more detailed analysis
            prompt = f"""
            As a senior AWS migration consultant with deep expertise in database migrations, provide a comprehensive analysis of this migration scenario:

            CURRENT INFRASTRUCTURE:
            - Source Database: {config['source_database_engine']} ({config['database_size_gb']} GB)
            - Target Database: {config['database_engine']}
            - Operating System: {config['operating_system']}
            - Platform: {config['server_type']}
            - Hardware: {config['cpu_cores']} cores @ {config['cpu_ghz']} GHz, {config['ram_gb']} GB RAM
            - Network: {config['nic_type']} ({config['nic_speed']} Mbps)
            - Environment: {config['environment']}
            - Performance Requirement: {config['performance_requirements']}
            - Downtime Tolerance: {config['downtime_tolerance_minutes']} minutes
            - Migration Agents: {config.get('number_of_agents', 1)} agents configured
            - Destination Storage: {config.get('destination_storage_type', 'S3')}

            CURRENT PERFORMANCE METRICS:
            - Database TPS: {performance_data.get('database_performance', {}).get('effective_tps', 'Unknown')}
            - Storage IOPS: {performance_data.get('storage_performance', {}).get('effective_iops', 'Unknown')}
            - Network Bandwidth: {performance_data.get('network_performance', {}).get('effective_bandwidth_mbps', 'Unknown')} Mbps
            - OS Efficiency: {performance_data.get('os_impact', {}).get('total_efficiency', 0) * 100:.1f}%
            - Overall Performance Score: {performance_data.get('performance_score', 0):.1f}/100

            Please provide a detailed assessment including:

            1. MIGRATION COMPLEXITY (1-10 scale with detailed justification)
            2. RISK ASSESSMENT with specific risk percentages and mitigation strategies
            3. PERFORMANCE OPTIMIZATION recommendations with expected improvement percentages
            4. DETAILED TIMELINE with phase-by-phase breakdown
            5. RESOURCE ALLOCATION with specific AWS instance recommendations
            6. COST OPTIMIZATION strategies with potential savings
            7. BEST PRACTICES specific to this configuration with implementation steps
            8. TESTING STRATEGY with checkpoints and validation criteria
            9. ROLLBACK PROCEDURES and contingency planning
            10. POST-MIGRATION monitoring and optimization recommendations
            11. AGENT SCALING IMPACT analysis based on {config.get('number_of_agents', 1)} agents
            12. DESTINATION STORAGE IMPACT for {config.get('destination_storage_type', 'S3')} including performance and cost implications

            Provide quantitative analysis wherever possible, including specific metrics, percentages, and measurable outcomes.
            Format the response as detailed sections with clear recommendations and actionable insights.
            """
            
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse AI response
            ai_response = message.content[0].text
            
            # Enhanced parsing for detailed analysis
            ai_analysis = self._parse_detailed_ai_response(ai_response, config, performance_data)
            
            return {
                'ai_complexity_score': ai_analysis.get('complexity_score', 6),
                'risk_factors': ai_analysis.get('risk_factors', []),
                'risk_percentages': ai_analysis.get('risk_percentages', {}),
                'mitigation_strategies': ai_analysis.get('mitigation_strategies', []),
                'performance_recommendations': ai_analysis.get('performance_recommendations', []),
                'performance_improvements': ai_analysis.get('performance_improvements', {}),
                'timeline_suggestions': ai_analysis.get('timeline_suggestions', []),
                'resource_allocation': ai_analysis.get('resource_allocation', {}),
                'cost_optimization': ai_analysis.get('cost_optimization', []),
                'best_practices': ai_analysis.get('best_practices', []),
                'testing_strategy': ai_analysis.get('testing_strategy', []),
                'rollback_procedures': ai_analysis.get('rollback_procedures', []),
                'post_migration_monitoring': ai_analysis.get('post_migration_monitoring', []),
                'confidence_level': ai_analysis.get('confidence_level', 'medium'),
                'detailed_assessment': ai_analysis.get('detailed_assessment', {}),
                'agent_scaling_impact': ai_analysis.get('agent_scaling_impact', {}),
                'destination_storage_impact': ai_analysis.get('destination_storage_impact', {}),
                'raw_ai_response': ai_response
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            st.error(f"AI Analysis Error: {str(e)}")
            return self._fallback_workload_analysis(config, performance_data)
    
    def _parse_detailed_ai_response(self, ai_response: str, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced parsing for detailed AI analysis"""
        
        # Calculate complexity score based on multiple factors
        complexity_factors = []
        base_complexity = 5
        
        # Database engine complexity
        if config['source_database_engine'] != config['database_engine']:
            complexity_factors.append(('Heterogeneous migration', 2))
            base_complexity += 2
        
        # Database size complexity
        if config['database_size_gb'] > 10000:
            complexity_factors.append(('Large database size', 1.5))
            base_complexity += 1.5
        elif config['database_size_gb'] > 5000:
            complexity_factors.append(('Medium database size', 0.5))
            base_complexity += 0.5
        
        # Performance requirements
        if config['performance_requirements'] == 'high':
            complexity_factors.append(('High performance requirements', 1))
            base_complexity += 1
        
        # Environment complexity
        if config['environment'] == 'production':
            complexity_factors.append(('Production environment', 0.5))
            base_complexity += 0.5
        
        # OS complexity
        if 'windows' in config['operating_system']:
            complexity_factors.append(('Windows licensing considerations', 0.5))
            base_complexity += 0.5
        
        # Downtime constraints
        if config['downtime_tolerance_minutes'] < 60:
            complexity_factors.append(('Strict downtime requirements', 1))
            base_complexity += 1
        
        # Agent scaling impact
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 3:
            complexity_factors.append(('Multi-agent coordination complexity', 0.5))
            base_complexity += 0.5
        elif num_agents == 1:
            complexity_factors.append(('Single agent bottleneck risk', 0.3))
            base_complexity += 0.3
        
        # Destination storage complexity
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            complexity_factors.append(('FSx for Windows File System complexity', 0.8))
            base_complexity += 0.8
        elif destination_storage == 'FSx_Lustre':
            complexity_factors.append(('FSx for Lustre high-performance complexity', 1.0))
            base_complexity += 1.0
        
        complexity_score = min(10, max(1, base_complexity))
        
        # Generate detailed risk assessment
        risk_factors = []
        risk_percentages = {}
        
        if config['source_database_engine'] != config['database_engine']:
            risk_factors.append("Schema conversion complexity may cause compatibility issues")
            risk_percentages['schema_conversion_risk'] = 25
        
        if config['database_size_gb'] > 5000:
            risk_factors.append("Large database size increases migration time and failure probability")
            risk_percentages['large_database_risk'] = 15
        
        if config['downtime_tolerance_minutes'] < 120:
            risk_factors.append("Tight downtime window may require multiple attempts")
            risk_percentages['downtime_risk'] = 20
        
        # Agent-specific risks
        if num_agents == 1 and config['database_size_gb'] > 5000:
            risk_factors.append("Single agent may become throughput bottleneck")
            risk_percentages['agent_bottleneck_risk'] = 30
        elif num_agents > 5:
            risk_factors.append("Complex multi-agent coordination may cause synchronization issues")
            risk_percentages['coordination_risk'] = 15
        
        # Destination storage risks
        if destination_storage == 'FSx_Windows':
            risk_factors.append("FSx for Windows may require additional Active Directory integration")
            risk_percentages['fsx_windows_risk'] = 10
        elif destination_storage == 'FSx_Lustre':
            risk_factors.append("FSx for Lustre high-performance configuration complexity")
            risk_percentages['fsx_lustre_risk'] = 15
        
        perf_score = performance_data.get('performance_score', 0)
        if perf_score < 70:
            risk_factors.append("Current performance issues may impact migration success")
            risk_percentages['performance_risk'] = 30
        
        # Generate detailed recommendations
        performance_recommendations = []
        performance_improvements = {}
        
        if perf_score < 80:
            performance_recommendations.append("Optimize database queries and indexes before migration")
            performance_improvements['query_optimization'] = '15-25%'
        
        if config['ram_gb'] < 32:
            performance_recommendations.append("Consider memory upgrade for better migration throughput")
            performance_improvements['memory_upgrade'] = '20-30%'
        
        # Agent scaling recommendations
        if num_agents > 1:
            performance_recommendations.append(f"Optimize {num_agents} agents for parallel processing")
            performance_improvements['agent_scaling'] = f'{min(num_agents * 15, 75)}%'
        
        # Destination storage recommendations
        if destination_storage == 'FSx_Lustre':
            performance_recommendations.append("Leverage FSx for Lustre high-performance capabilities")
            performance_improvements['fsx_lustre_performance'] = '40-60%'
        elif destination_storage == 'FSx_Windows':
            performance_recommendations.append("Optimize FSx for Windows file sharing protocols")
            performance_improvements['fsx_windows_optimization'] = '20-35%'
        
        # Enhanced timeline with phases
        timeline_suggestions = [
            "Phase 1: Assessment and Planning (2-3 weeks)",
            "Phase 2: Environment Setup and Testing (2-4 weeks)", 
            "Phase 3: Data Validation and Performance Testing (1-2 weeks)",
            "Phase 4: Migration Execution (1-3 days)",
            "Phase 5: Post-Migration Validation and Optimization (1 week)"
        ]
        
        # Detailed resource allocation with agent scaling considerations
        resource_allocation = {
            'migration_team_size': 3 + (complexity_score // 3) + (num_agents // 3),
            'aws_specialists_needed': 1 if complexity_score < 6 else 2,
            'database_experts_required': 1 if config['source_database_engine'] == config['database_engine'] else 2,
            'testing_resources': '2-3 dedicated testers for ' + ('2 weeks' if complexity_score < 7 else '3-4 weeks'),
            'infrastructure_requirements': f"Staging environment with {config['cpu_cores']*2} cores and {config['ram_gb']*1.5} GB RAM",
            'agent_management_overhead': f"{num_agents} agents require {max(1, num_agents // 2)} dedicated monitoring specialists",
            'storage_specialists': 1 if destination_storage != 'S3' else 0
        }
        
        # Agent scaling impact analysis
        agent_scaling_impact = {
            'parallel_processing_benefit': min(num_agents * 20, 80),  # Diminishing returns
            'coordination_overhead': max(0, (num_agents - 1) * 5),
            'throughput_multiplier': min(num_agents * 0.8, 4.0),  # Not linear scaling
            'management_complexity': num_agents * 10,
            'optimal_agent_count': self._calculate_optimal_agents(config),
            'current_efficiency': min(100, (100 - (abs(num_agents - self._calculate_optimal_agents(config)) * 10)))
        }
        
        # Destination storage impact analysis
        destination_storage_impact = {
            'storage_type': destination_storage,
            'performance_impact': self._calculate_storage_performance_impact(destination_storage),
            'cost_impact': self._calculate_storage_cost_impact(destination_storage),
            'complexity_factor': self._get_storage_complexity_factor(destination_storage),
            'recommended_for': self._get_storage_recommendations(destination_storage, config)
        }
        
        return {
            'complexity_score': complexity_score,
            'complexity_factors': complexity_factors,
            'risk_factors': risk_factors,
            'risk_percentages': risk_percentages,
            'mitigation_strategies': self._generate_mitigation_strategies(risk_factors, config),
            'performance_recommendations': performance_recommendations,
            'performance_improvements': performance_improvements,
            'timeline_suggestions': timeline_suggestions,
            'resource_allocation': resource_allocation,
            'cost_optimization': self._generate_cost_optimization(config, complexity_score),
            'best_practices': self._generate_best_practices(config, complexity_score),
            'testing_strategy': self._generate_testing_strategy(config, complexity_score),
            'rollback_procedures': self._generate_rollback_procedures(config),
            'post_migration_monitoring': self._generate_monitoring_recommendations(config),
            'confidence_level': 'high' if complexity_score < 6 else 'medium' if complexity_score < 8 else 'requires_specialist_review',
            'agent_scaling_impact': agent_scaling_impact,
            'destination_storage_impact': destination_storage_impact,
            'detailed_assessment': {
                'overall_readiness': 'ready' if perf_score > 75 and complexity_score < 7 else 'needs_preparation' if perf_score > 60 else 'significant_preparation_required',
                'success_probability': max(60, 95 - (complexity_score * 5) - max(0, (70 - perf_score)) + (agent_scaling_impact['current_efficiency'] // 10)),
                'recommended_approach': 'direct_migration' if complexity_score < 6 and config['database_size_gb'] < 2000 else 'staged_migration',
                'critical_success_factors': self._identify_critical_success_factors(config, complexity_score)
            }
        }
    
    def _calculate_optimal_agents(self, config: Dict) -> int:
        """Calculate optimal number of agents based on database size and requirements"""
        database_size = config['database_size_gb']
        
        if database_size < 1000:
            return 1
        elif database_size < 5000:
            return 2
        elif database_size < 20000:
            return 3
        elif database_size < 50000:
            return 4
        else:
            return 5
    
    def _calculate_storage_performance_impact(self, storage_type: str) -> Dict:
        """Calculate performance impact for different storage destinations"""
        storage_profiles = {
            'S3': {
                'throughput_multiplier': 1.0,
                'latency_impact': 1.0,
                'iops_capability': 'Standard',
                'performance_rating': 'Good'
            },
            'FSx_Windows': {
                'throughput_multiplier': 1.3,
                'latency_impact': 0.7,
                'iops_capability': 'High',
                'performance_rating': 'Very Good'
            },
            'FSx_Lustre': {
                'throughput_multiplier': 2.0,
                'latency_impact': 0.4,
                'iops_capability': 'Very High',
                'performance_rating': 'Excellent'
            }
        }
        return storage_profiles.get(storage_type, storage_profiles['S3'])
    
    def _calculate_storage_cost_impact(self, storage_type: str) -> Dict:
        """Calculate cost impact for different storage destinations"""
        cost_profiles = {
            'S3': {
                'base_cost_multiplier': 1.0,
                'operational_cost': 'Low',
                'setup_cost': 'Minimal',
                'long_term_value': 'Excellent'
            },
            'FSx_Windows': {
                'base_cost_multiplier': 2.5,
                'operational_cost': 'Medium',
                'setup_cost': 'Medium',
                'long_term_value': 'Good'
            },
            'FSx_Lustre': {
                'base_cost_multiplier': 4.0,
                'operational_cost': 'High',
                'setup_cost': 'High',
                'long_term_value': 'Good for HPC'
            }
        }
        return cost_profiles.get(storage_type, cost_profiles['S3'])
    
    def _get_storage_complexity_factor(self, storage_type: str) -> float:
        """Get complexity factor for storage type"""
        complexity_factors = {
            'S3': 1.0,
            'FSx_Windows': 1.8,
            'FSx_Lustre': 2.2
        }
        return complexity_factors.get(storage_type, 1.0)
    
    def _get_storage_recommendations(self, storage_type: str, config: Dict) -> List[str]:
        """Get recommendations for storage type"""
        recommendations = {
            'S3': [
                "Cost-effective for most workloads",
                "Excellent durability and availability",
                "Simple integration with migration tools",
                "Automatic scaling and management"
            ],
            'FSx_Windows': [
                "Ideal for Windows-based applications",
                "Native Windows file system features",
                "Active Directory integration",
                "Better performance for file-based workloads"
            ],
            'FSx_Lustre': [
                "Best for high-performance computing",
                "Extremely high throughput and IOPS",
                "Optimized for parallel processing",
                "Ideal for analytics and ML workloads"
            ]
        }
        return recommendations.get(storage_type, recommendations['S3'])
    
    def _generate_mitigation_strategies(self, risk_factors: List[str], config: Dict) -> List[str]:
        """Generate specific mitigation strategies"""
        strategies = []
        
        if any('schema' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct comprehensive schema conversion testing with AWS SCT")
            strategies.append("Create detailed schema mapping documentation")
            strategies.append("Implement phased schema migration with rollback checkpoints")
        
        if any('database size' in risk.lower() for risk in risk_factors):
            strategies.append("Implement parallel data transfer using multiple DMS tasks")
            strategies.append("Use AWS DataSync for initial bulk data transfer")
            strategies.append("Schedule migration during low-traffic periods")
        
        if any('downtime' in risk.lower() for risk in risk_factors):
            strategies.append("Implement read replica for near-zero downtime migration")
            strategies.append("Use AWS DMS ongoing replication for data synchronization")
            strategies.append("Prepare detailed rollback procedures with time estimates")
        
        if any('performance' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct pre-migration performance tuning")
            strategies.append("Implement AWS CloudWatch monitoring throughout migration")
            strategies.append("Establish performance baselines and acceptance criteria")
        
        if any('agent' in risk.lower() for risk in risk_factors):
            strategies.append(f"Optimize {config.get('number_of_agents', 1)} agent configuration for workload")
            strategies.append("Implement agent health monitoring and automatic failover")
            strategies.append("Configure load balancing across multiple agents")
        
        if any('fsx' in risk.lower() for risk in risk_factors):
            strategies.append("Test FSx integration thoroughly in staging environment")
            strategies.append("Implement FSx performance monitoring and optimization")
            strategies.append("Plan for FSx-specific backup and recovery procedures")
        
        return strategies
    
    def _generate_cost_optimization(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate cost optimization strategies"""
        optimizations = []
        
        if config['database_size_gb'] < 1000:
            optimizations.append("Consider Reserved Instances for 20-30% cost savings")
        
        if config['environment'] == 'non-production':
            optimizations.append("Use Spot Instances for development/testing to reduce costs by 60-70%")
        
        if complexity_score < 6:
            optimizations.append("Leverage AWS Managed Services (RDS) to reduce operational overhead")
        
        optimizations.append("Implement automated scaling policies to optimize resource utilization")
        
        # Storage-specific optimizations
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'S3':
            optimizations.append("Use S3 Intelligent Tiering for backup storage cost optimization")
        elif destination_storage == 'FSx_Windows':
            optimizations.append("Right-size FSx for Windows based on actual usage patterns")
        elif destination_storage == 'FSx_Lustre':
            optimizations.append("Use FSx for Lustre scratch file systems for temporary high-performance needs")
        
        # Agent-specific optimizations
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 3:
            optimizations.append("Consider agent consolidation to reduce licensing and management costs")
        elif num_agents == 1 and config['database_size_gb'] > 5000:
            optimizations.append("Scale up to multiple agents for faster migration and reduced window costs")
        
        return optimizations
    
    def _generate_best_practices(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate specific best practices"""
        practices = []
        
        practices.append("Implement comprehensive backup strategy before migration initiation")
        practices.append("Use AWS Migration Hub for centralized migration tracking")
        practices.append("Establish detailed communication plan with stakeholders")
        
        if config['database_engine'] in ['mysql', 'postgresql']:
            practices.append("Leverage native database replication for minimal downtime")
        
        if complexity_score > 7:
            practices.append("Engage AWS Professional Services for complex migration scenarios")
        
        practices.append("Implement automated testing pipelines for validation")
        practices.append("Create detailed runbook with step-by-step procedures")
        
        # Agent-specific best practices
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 1:
            practices.append(f"Configure {num_agents} agents with proper load distribution")
            practices.append("Implement centralized agent monitoring and logging")
            practices.append("Test agent failover scenarios during non-production phases")
        
        # Storage-specific best practices
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            practices.append("Configure FSx for Windows with appropriate backup policies")
            practices.append("Implement proper Active Directory integration for FSx")
        elif destination_storage == 'FSx_Lustre':
            practices.append("Optimize FSx for Lustre configuration for specific workload patterns")
            practices.append("Plan for FSx for Lustre data repository associations")
        
        return practices
    
    def _generate_testing_strategy(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate comprehensive testing strategy"""
        strategy = []
        
        strategy.append("Unit Testing: Validate individual migration components")
        strategy.append("Integration Testing: Test end-to-end migration workflow")
        strategy.append("Performance Testing: Validate AWS environment performance")
        strategy.append("Data Integrity Testing: Verify data consistency and completeness")
        
        if config['source_database_engine'] != config['database_engine']:
            strategy.append("Schema Conversion Testing: Validate converted database objects")
            strategy.append("Application Compatibility Testing: Ensure application functionality")
        
        strategy.append("Disaster Recovery Testing: Validate backup and restore procedures")
        strategy.append("Security Testing: Verify access controls and encryption")
        
        # Agent-specific testing
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 1:
            strategy.append(f"Multi-Agent Testing: Validate {num_agents} agent coordination")
            strategy.append("Load Balancing Testing: Verify even distribution across agents")
        
        # Storage-specific testing
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage != 'S3':
            strategy.append(f"FSx Performance Testing: Validate {destination_storage} performance characteristics")
            strategy.append(f"FSx Integration Testing: Test application connectivity to {destination_storage}")
        
        return strategy
    
    def _generate_rollback_procedures(self, config: Dict) -> List[str]:
        """Generate rollback procedures"""
        procedures = []
        
        procedures.append("Maintain synchronized read replica during migration window")
        procedures.append("Create point-in-time recovery snapshot before cutover")
        procedures.append("Prepare DNS switching procedures for quick rollback")
        procedures.append("Document application configuration rollback steps")
        procedures.append("Establish go/no-go decision criteria with specific metrics")
        procedures.append("Test rollback procedures in staging environment")
        
        # Agent-specific rollback procedures
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 1:
            procedures.append(f"Coordinate {num_agents} agent shutdown procedures")
            procedures.append("Implement agent state synchronization for rollback")
        
        # Storage-specific rollback
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage != 'S3':
            procedures.append(f"Prepare {destination_storage} rollback and data recovery procedures")
            procedures.append(f"Test {destination_storage} backup and restore capabilities")
        
        return procedures
    
    def _generate_monitoring_recommendations(self, config: Dict) -> List[str]:
        """Generate post-migration monitoring recommendations"""
        monitoring = []
        
        monitoring.append("Implement CloudWatch detailed monitoring for all database metrics")
        monitoring.append("Set up automated alerts for performance degradation")
        monitoring.append("Monitor application response times and error rates")
        monitoring.append("Track database connection patterns and query performance")
        monitoring.append("Implement cost monitoring and optimization alerts")
        monitoring.append("Schedule regular performance reviews for first 30 days")
        
        # Agent-specific monitoring
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 1:
            monitoring.append(f"Monitor {num_agents} agent performance and health metrics")
            monitoring.append("Track agent load distribution and efficiency")
        
        # Storage-specific monitoring
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            monitoring.append("Monitor FSx for Windows file system performance and utilization")
            monitoring.append("Track Windows file sharing protocol efficiency")
        elif destination_storage == 'FSx_Lustre':
            monitoring.append("Monitor FSx for Lustre high-performance metrics")
            monitoring.append("Track parallel processing efficiency and throughput")
        
        return monitoring
    
    def _identify_critical_success_factors(self, config: Dict, complexity_score: int) -> List[str]:
        """Identify critical success factors"""
        factors = []
        
        factors.append("Stakeholder alignment on migration timeline and expectations")
        factors.append("Comprehensive testing in staging environment matching production")
        factors.append("Skilled migration team with AWS and database expertise")
        
        if complexity_score > 7:
            factors.append("Dedicated AWS solutions architect involvement")
        
        if config['downtime_tolerance_minutes'] < 120:
            factors.append("Near-zero downtime migration strategy implementation")
        
        factors.append("Robust monitoring and alerting throughout migration process")
        factors.append("Clear rollback criteria and tested rollback procedures")
        
        # Agent-specific success factors
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 2:
            factors.append(f"Proper coordination and management of {num_agents} migration agents")
            factors.append("Agent performance optimization and load balancing")
        
        # Storage-specific success factors
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage != 'S3':
            factors.append(f"Successful {destination_storage} integration and performance validation")
            factors.append(f"Proper {destination_storage} configuration and optimization")
        
        return factors
    
    def _fallback_workload_analysis(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced fallback analysis when AI is not available"""
        
        # Calculate complexity based on configuration
        complexity_score = 5
        if config['source_database_engine'] != config['database_engine']:
            complexity_score += 2
        if config['database_size_gb'] > 5000:
            complexity_score += 1
        if config['performance_requirements'] == 'high':
            complexity_score += 1
        if config['downtime_tolerance_minutes'] < 60:
            complexity_score += 1
        
        # Storage complexity
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            complexity_score += 0.8
        elif destination_storage == 'FSx_Lustre':
            complexity_score += 1.0
        
        # Agent scaling impact
        num_agents = config.get('number_of_agents', 1)
        optimal_agents = self._calculate_optimal_agents(config)
        agent_efficiency = min(100, (100 - (abs(num_agents - optimal_agents) * 10)))
        
        return {
            'ai_complexity_score': min(10, complexity_score),
            'risk_factors': [
                "Migration complexity varies with database engine differences",
                "Large database sizes increase migration duration",
                "Performance requirements may necessitate additional testing",
                f"Agent configuration ({num_agents} agents) may impact throughput",
                f"Destination storage ({destination_storage}) affects performance and complexity"
            ],
            'mitigation_strategies': [
                "Conduct thorough pre-migration testing",
                "Plan for adequate migration windows",
                "Implement comprehensive backup strategies",
                f"Optimize {num_agents} agent configuration for workload",
                f"Validate {destination_storage} integration thoroughly"
            ],
            'performance_recommendations': [
                "Optimize database before migration",
                "Consider read replicas for minimal downtime",
                "Monitor performance throughout migration",
                f"Fine-tune {num_agents} agent performance settings",
                f"Leverage {destination_storage} performance characteristics"
            ],
            'confidence_level': 'medium',
            'agent_scaling_impact': {
                'current_efficiency': agent_efficiency,
                'optimal_agent_count': optimal_agents,
                'throughput_multiplier': min(num_agents * 0.8, 4.0)
            },
            'destination_storage_impact': {
                'storage_type': destination_storage,
                'performance_impact': self._calculate_storage_performance_impact(destination_storage),
                'cost_impact': self._calculate_storage_cost_impact(destination_storage),
                'complexity_factor': self._get_storage_complexity_factor(destination_storage)
            },
            'resource_allocation': {
                'migration_team_size': 3 + (num_agents // 2),
                'aws_specialists_needed': 1,
                'database_experts_required': 1,
                'storage_specialists': 1 if destination_storage != 'S3' else 0
            },
            'detailed_assessment': {
                'success_probability': max(60, 85 - complexity_score * 3 + (agent_efficiency // 10))
            },
            'raw_ai_response': 'AI analysis not available - using fallback analysis'
        }

class AWSAPIManager:
    """Manage AWS API integration for real-time pricing and optimization"""
    
    def __init__(self):
        self.session = None
        self.pricing_client = None
        self.compute_optimizer_client = None
        self.connected = False
        
        try:
            # Try to initialize AWS session
            self.session = boto3.Session()
            self.pricing_client = self.session.client('pricing', region_name='us-east-1')
            
            # Test connection
            self.pricing_client.describe_services(MaxResults=1)
            self.connected = True
            logger.info("AWS API clients initialized successfully")
            
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"AWS API initialization failed: {e}")
            self.connected = False
    
    async def get_real_time_pricing(self, region: str = 'us-west-2') -> Dict:
        """Fetch real-time AWS pricing data including FSx"""
        if not self.connected:
            return self._fallback_pricing_data(region)
        
        try:
            # Get EC2 pricing
            ec2_pricing = await self._get_ec2_pricing(region)
            
            # Get RDS pricing
            rds_pricing = await self._get_rds_pricing(region)
            
            # Get storage pricing (S3, EBS)
            storage_pricing = await self._get_storage_pricing(region)
            
            # Get FSx pricing
            fsx_pricing = await self._get_fsx_pricing(region)
            
            return {
                'region': region,
                'last_updated': datetime.now(),
                'ec2_instances': ec2_pricing,
                'rds_instances': rds_pricing,
                'storage': storage_pricing,
                'fsx': fsx_pricing,
                'data_source': 'aws_api'
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch AWS pricing: {e}")
            return self._fallback_pricing_data(region)
    
    async def _get_fsx_pricing(self, region: str) -> Dict:
        """Get FSx pricing for Windows and Lustre"""
        try:
            fsx_pricing = {}
            
            # FSx for Windows File Server
            try:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonFSx',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'fileSystemType', 'Value': 'Windows'},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                        {'Type': 'TERM_MATCH', 'Field': 'storageType', 'Value': 'SSD'}
                    ],
                    MaxResults=1
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        term_data = list(terms.values())[0]
                        price_dimensions = term_data.get('priceDimensions', {})
                        if price_dimensions:
                            price_info = list(price_dimensions.values())[0]
                            price_per_gb_month = float(price_info['pricePerUnit']['USD'])
                            
                            fsx_pricing['windows'] = {
                                'price_per_gb_month': price_per_gb_month,
                                'minimum_size_gb': 32,
                                'maximum_size_gb': 65536,
                                'throughput_capacity_mbps': [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                                'backup_retention': True,
                                'multi_az': True
                            }
                            
            except Exception as e:
                logger.warning(f"Failed to get FSx Windows pricing: {e}")
                fsx_pricing['windows'] = self._get_fallback_fsx_windows_pricing()
            
            # FSx for Lustre
            try:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonFSx',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'fileSystemType', 'Value': 'Lustre'},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                        {'Type': 'TERM_MATCH', 'Field': 'storageType', 'Value': 'SSD'}
                    ],
                    MaxResults=1
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        term_data = list(terms.values())[0]
                        price_dimensions = term_data.get('priceDimensions', {})
                        if price_dimensions:
                            price_info = list(price_dimensions.values())[0]
                            price_per_gb_month = float(price_info['pricePerUnit']['USD'])
                            
                            fsx_pricing['lustre'] = {
                                'price_per_gb_month': price_per_gb_month,
                                'minimum_size_gb': 1200,
                                'maximum_size_gb': 100800,
                                'throughput_per_tib': [50, 100, 200],
                                'deployment_type': ['SCRATCH_1', 'SCRATCH_2', 'PERSISTENT_1', 'PERSISTENT_2'],
                                'data_repository_association': True
                            }
                            
            except Exception as e:
                logger.warning(f"Failed to get FSx Lustre pricing: {e}")
                fsx_pricing['lustre'] = self._get_fallback_fsx_lustre_pricing()
            
            return fsx_pricing
            
        except Exception as e:
            logger.error(f"FSx pricing fetch failed: {e}")
            return self._fallback_fsx_pricing()
    
    async def _get_ec2_pricing(self, region: str) -> Dict:
        """Get EC2 instance pricing"""
        try:
            # AWS Pricing API calls for EC2 instances
            instance_types = ['t3.medium', 't3.large', 't3.xlarge', 'c5.large', 'c5.xlarge', 
                            'c5.2xlarge', 'c5.4xlarge', 'r6i.large', 'r6i.xlarge', 
                            'r6i.2xlarge', 'r6i.4xlarge', 'r6i.8xlarge']
            
            pricing_data = {}
            
            for instance_type in instance_types:
                try:
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonEC2',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                            {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        # Extract pricing information
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                price_per_hour = float(price_info['pricePerUnit']['USD'])
                                
                                # Get instance specs
                                attributes = price_data.get('product', {}).get('attributes', {})
                                
                                pricing_data[instance_type] = {
                                    'vcpu': int(attributes.get('vcpu', 2)),
                                    'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                    'cost_per_hour': price_per_hour
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get pricing for {instance_type}: {e}")
                    # Use fallback pricing
                    pricing_data[instance_type] = self._get_fallback_instance_pricing(instance_type)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"EC2 pricing fetch failed: {e}")
            return self._fallback_ec2_pricing()
    
    async def _get_rds_pricing(self, region: str) -> Dict:
        """Get RDS instance pricing"""
        try:
            instance_types = ['db.t3.medium', 'db.t3.large', 'db.r6g.large', 'db.r6g.xlarge', 
                            'db.r6g.2xlarge', 'db.r6g.4xlarge', 'db.r6g.8xlarge']
            
            pricing_data = {}
            
            for instance_type in instance_types:
                try:
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonRDS',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                            {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': 'MySQL'},
                            {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                price_per_hour = float(price_info['pricePerUnit']['USD'])
                                
                                attributes = price_data.get('product', {}).get('attributes', {})
                                
                                pricing_data[instance_type] = {
                                    'vcpu': int(attributes.get('vcpu', 2)),
                                    'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                    'cost_per_hour': price_per_hour
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get RDS pricing for {instance_type}: {e}")
                    pricing_data[instance_type] = self._get_fallback_rds_pricing(instance_type)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"RDS pricing fetch failed: {e}")
            return self._fallback_rds_pricing()
    
    async def _get_storage_pricing(self, region: str) -> Dict:
        """Get EBS and S3 storage pricing"""
        try:
            storage_types = ['gp3', 'io1', 'io2']
            pricing_data = {}
            
            for storage_type in storage_types:
                try:
                    volume_type_map = {
                        'gp3': 'General Purpose',
                        'io1': 'Provisioned IOPS',
                        'io2': 'Provisioned IOPS'
                    }
                    
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonEC2',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'Storage'},
                            {'Type': 'TERM_MATCH', 'Field': 'volumeType', 'Value': volume_type_map.get(storage_type, 'General Purpose')},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                price_per_gb = float(price_info['pricePerUnit']['USD'])
                                
                                pricing_data[storage_type] = {
                                    'cost_per_gb_month': price_per_gb,
                                    'iops_included': 3000 if storage_type == 'gp3' else 0,
                                    'cost_per_iops_month': 0.065 if storage_type in ['io1', 'io2'] else 0
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get storage pricing for {storage_type}: {e}")
                    pricing_data[storage_type] = self._get_fallback_storage_pricing(storage_type)
            
            # Add S3 pricing
            pricing_data['s3_standard'] = {
                'cost_per_gb_month': 0.023,
                'requests_per_1000': 0.0004,
                'data_transfer_out_per_gb': 0.09
            }
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Storage pricing fetch failed: {e}")
            return self._fallback_storage_pricing()
    
    def _region_to_location(self, region: str) -> str:
        """Convert AWS region to location name for pricing API"""
        region_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)'
        }
        return region_map.get(region, 'US West (Oregon)')
    
    def _extract_memory_gb(self, memory_str: str) -> int:
        """Extract memory in GB from AWS memory string"""
        try:
            # Extract number from strings like "4 GiB", "8.0 GiB"
            import re
            match = re.search(r'([\d.]+)', memory_str)
            if match:
                return int(float(match.group(1)))
            return 4  # Default
        except:
            return 4
    
    def _fallback_pricing_data(self, region: str) -> Dict:
        """Fallback pricing data when API is unavailable"""
        return {
            'region': region,
            'last_updated': datetime.now(),
            'data_source': 'fallback',
            'ec2_instances': self._fallback_ec2_pricing(),
            'rds_instances': self._fallback_rds_pricing(),
            'storage': self._fallback_storage_pricing(),
            'fsx': self._fallback_fsx_pricing()
        }
    
    def _fallback_ec2_pricing(self) -> Dict:
        """Fallback EC2 pricing data"""
        return {
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
        }
    
    def _fallback_rds_pricing(self) -> Dict:
        """Fallback RDS pricing data"""
        return {
            'db.t3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.068},
            'db.t3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.136},
            'db.r6g.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.48},
            'db.r6g.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.96},
            'db.r6g.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.92},
            'db.r6g.4xlarge': {'vcpu': 16, 'memory': 128, 'cost_per_hour': 3.84},
            'db.r6g.8xlarge': {'vcpu': 32, 'memory': 256, 'cost_per_hour': 7.68}
        }
    
    def _fallback_storage_pricing(self) -> Dict:
        """Fallback storage pricing data"""
        return {
            'gp3': {'cost_per_gb_month': 0.08, 'iops_included': 3000},
            'io1': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065},
            'io2': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065},
            's3_standard': {
                'cost_per_gb_month': 0.023,
                'requests_per_1000': 0.0004,
                'data_transfer_out_per_gb': 0.09
            }
        }
    
    def _fallback_fsx_pricing(self) -> Dict:
        """Fallback FSx pricing data"""
        return {
            'windows': self._get_fallback_fsx_windows_pricing(),
            'lustre': self._get_fallback_fsx_lustre_pricing()
        }
    
    def _get_fallback_fsx_windows_pricing(self) -> Dict:
        """Get fallback FSx Windows pricing"""
        return {
            'price_per_gb_month': 0.13,
            'minimum_size_gb': 32,
            'maximum_size_gb': 65536,
            'throughput_capacity_mbps': [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            'backup_retention': True,
            'multi_az': True
        }
    
    def _get_fallback_fsx_lustre_pricing(self) -> Dict:
        """Get fallback FSx Lustre pricing"""
        return {
            'price_per_gb_month': 0.14,
            'minimum_size_gb': 1200,
            'maximum_size_gb': 100800,
            'throughput_per_tib': [50, 100, 200],
            'deployment_type': ['SCRATCH_1', 'SCRATCH_2', 'PERSISTENT_1', 'PERSISTENT_2'],
            'data_repository_association': True
        }
    
    def _get_fallback_instance_pricing(self, instance_type: str) -> Dict:
        """Get fallback pricing for specific instance type"""
        fallback_data = self._fallback_ec2_pricing()
        return fallback_data.get(instance_type, {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.05})
    
    def _get_fallback_rds_pricing(self, instance_type: str) -> Dict:
        """Get fallback RDS pricing for specific instance type"""
        fallback_data = self._fallback_rds_pricing()
        return fallback_data.get(instance_type, {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.07})
    
    def _get_fallback_storage_pricing(self, storage_type: str) -> Dict:
        """Get fallback storage pricing for specific type"""
        fallback_data = self._fallback_storage_pricing()
        return fallback_data.get(storage_type, {'cost_per_gb_month': 0.08, 'iops_included': 0})

class OSPerformanceManager:
    """Enhanced OS performance manager with AI insights"""
    
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
                    'mysql': 0.88, 'postgresql': 0.85, 'oracle': 0.95, 'sqlserver': 0.98, 'mongodb': 0.87
                },
                'licensing_cost_factor': 2.5,
                'management_complexity': 0.6,
                'security_overhead': 0.08,
                'ai_insights': {
                    'strengths': ['Native SQL Server integration', 'Enterprise management tools', 'Active Directory integration'],
                    'weaknesses': ['Higher licensing costs', 'More resource overhead', 'Complex patch management'],
                    'migration_considerations': ['Licensing compliance', 'Service account migration', 'Registry dependencies']
                }
            },
            'windows_server_2022': {
                'name': 'Windows Server 2022',
                'cpu_efficiency': 0.95,
                'memory_efficiency': 0.92,
                'io_efficiency': 0.90,
                'network_efficiency': 0.93,
                'virtualization_overhead': 0.10,
                'database_optimizations': {
                    'mysql': 0.90, 'postgresql': 0.88, 'oracle': 0.97, 'sqlserver': 0.99, 'mongodb': 0.89
                },
                'licensing_cost_factor': 3.0,
                'management_complexity': 0.5,
                'security_overhead': 0.06,
                'ai_insights': {
                    'strengths': ['Improved container support', 'Enhanced security features', 'Better cloud integration'],
                    'weaknesses': ['Higher costs', 'Newer OS compatibility risks', 'Learning curve for admins'],
                    'migration_considerations': ['Hardware compatibility', 'Application compatibility testing', 'Training requirements']
                }
            },
            'rhel_8': {
                'name': 'Red Hat Enterprise Linux 8',
                'cpu_efficiency': 0.96,
                'memory_efficiency': 0.94,
                'io_efficiency': 0.95,
                'network_efficiency': 0.95,
                'virtualization_overhead': 0.06,
                'database_optimizations': {
                    'mysql': 0.95, 'postgresql': 0.97, 'oracle': 0.93, 'sqlserver': 0.85, 'mongodb': 0.96
                },
                'licensing_cost_factor': 1.5,
                'management_complexity': 0.7,
                'security_overhead': 0.04,
                'ai_insights': {
                    'strengths': ['Excellent performance', 'Strong container support', 'Comprehensive support'],
                    'weaknesses': ['Commercial licensing required', 'Steeper learning curve', 'Limited Windows application support'],
                    'migration_considerations': ['Staff training needs', 'Application compatibility', 'Support contracts']
                }
            },
            'rhel_9': {
                'name': 'Red Hat Enterprise Linux 9',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.96,
                'io_efficiency': 0.97,
                'network_efficiency': 0.97,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.97, 'postgresql': 0.98, 'oracle': 0.95, 'sqlserver': 0.87, 'mongodb': 0.97
                },
                'licensing_cost_factor': 1.8,
                'management_complexity': 0.6,
                'security_overhead': 0.03,
                'ai_insights': {
                    'strengths': ['Latest performance optimizations', 'Enhanced security', 'Modern container runtime'],
                    'weaknesses': ['Newer release stability', 'Application compatibility risks', 'Migration complexity'],
                    'migration_considerations': ['Extensive testing required', 'Legacy application assessment', 'Staff skill upgrade']
                }
            },
            'ubuntu_20_04': {
                'name': 'Ubuntu Server 20.04 LTS',
                'cpu_efficiency': 0.97,
                'memory_efficiency': 0.95,
                'io_efficiency': 0.96,
                'network_efficiency': 0.96,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.96, 'postgresql': 0.98, 'oracle': 0.90, 'sqlserver': 0.82, 'mongodb': 0.97
                },
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.8,
                'security_overhead': 0.03,
                'ai_insights': {
                    'strengths': ['No licensing costs', 'Great community support', 'Excellent for open source databases'],
                    'weaknesses': ['No commercial support without subscription', 'Requires Linux expertise', 'Less enterprise tooling'],
                    'migration_considerations': ['Staff Linux skills', 'Management tool migration', 'Support strategy']
                }
            },
            'ubuntu_22_04': {
                'name': 'Ubuntu Server 22.04 LTS',
                'cpu_efficiency': 0.98,
                'memory_efficiency': 0.97,
                'io_efficiency': 0.98,
                'network_efficiency': 0.98,
                'virtualization_overhead': 0.04,
                'database_optimizations': {
                    'mysql': 0.98, 'postgresql': 0.99, 'oracle': 0.92, 'sqlserver': 0.84, 'mongodb': 0.98
                },
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.7,
                'security_overhead': 0.02,
                'ai_insights': {
                    'strengths': ['Latest performance features', 'Enhanced security', 'Cloud-native optimizations'],
                    'weaknesses': ['Newer release risks', 'Potential compatibility issues', 'Learning curve'],
                    'migration_considerations': ['Comprehensive testing', 'Backup OS strategy', 'Monitoring setup']
                }
            }
        }
    
    def calculate_os_performance_impact(self, os_type: str, platform_type: str, database_engine: str) -> Dict:
        """Enhanced OS performance calculation with AI insights"""
        
        os_config = self.operating_systems[os_type]
        
        # Base OS efficiency calculation (preserved original logic)
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
            if 'windows' in os_type:
                total_efficiency *= 1.02
            else:
                total_efficiency *= 1.05
        
        # Enhanced return with AI insights
        return {
            **{k: v for k, v in os_config.items() if k != 'ai_insights'},
            'total_efficiency': total_efficiency,
            'base_efficiency': base_efficiency,
            'db_optimization': db_optimization,
            'virtualization_overhead': os_config['virtualization_overhead'] if platform_type == 'vmware' else 0,
            'ai_insights': os_config['ai_insights'],
            'platform_optimization': 1.02 if platform_type == 'physical' and 'windows' in os_type else 1.05 if platform_type == 'physical' else 1.0
        }

class EnhancedNetworkIntelligenceManager:
    """AI-powered network path intelligence with enhanced analysis including FSx destinations"""
    
    def __init__(self):
        self.network_paths = {
            'nonprod_sj_linux_nas_s3': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 15,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.92
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Linux NAS internal bandwidth', 'DX connection sharing'],
                    'optimization_opportunities': ['NAS performance tuning', 'DX bandwidth upgrade', 'Parallel transfer optimization'],
                    'risk_factors': ['Single DX connection dependency', 'NAS hardware limitations'],
                    'recommended_improvements': ['Implement NAS caching', 'Configure QoS on DX', 'Add backup connectivity']
                }
            },
            'nonprod_sj_linux_nas_fsx_windows': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS FSx for Windows',
                'destination_storage': 'FSx_Windows',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Windows',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS FSx Windows (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.5,
                        'ai_optimization_potential': 0.94
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Linux to Windows protocol conversion', 'SMB overhead'],
                    'optimization_opportunities': ['SMB3 protocol optimization', 'FSx throughput configuration', 'Connection pooling'],
                    'risk_factors': ['Cross-platform compatibility', 'SMB version negotiation'],
                    'recommended_improvements': ['Test SMB3.1.1 compatibility', 'Configure FSx performance mode', 'Implement connection caching']
                }
            },
            'nonprod_sj_linux_nas_fsx_lustre': {
                'name': 'Non-Prod: San Jose Linux NAS → AWS FSx for Lustre',
                'destination_storage': 'FSx_Lustre',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Lustre',
                'environment': 'non-production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 2,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.95
                    },
                    {
                        'name': 'Linux Jump Server to AWS FSx Lustre (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 8,
                        'reliability': 0.9995,
                        'connection_type': 'direct_connect',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.97
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Lustre client configuration', 'Parallel processing coordination'],
                    'optimization_opportunities': ['Lustre striping optimization', 'Parallel I/O tuning', 'Client-side caching'],
                    'risk_factors': ['Lustre complexity', 'Client compatibility'],
                    'recommended_improvements': ['Optimize Lustre striping patterns', 'Configure parallel data transfer', 'Tune Lustre clients']
                }
            },
            'nonprod_sj_windows_share_s3': {
                'name': 'Non-Prod: San Jose Windows Share → AWS S3',
                'destination_storage': 'S3',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 S3',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 3,
                        'reliability': 0.997,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.88
                    },
                    {
                        'name': 'Windows Jump Server to AWS S3 (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 18,
                        'reliability': 0.998,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.0,
                        'ai_optimization_potential': 0.90
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows file sharing overhead', 'SMB protocol latency', 'Windows network stack'],
                    'optimization_opportunities': ['SMB3 optimization', 'Windows TCP tuning', 'Antivirus exclusions'],
                    'risk_factors': ['Windows update interruptions', 'SMB vulnerabilities', 'File locking issues'],
                    'recommended_improvements': ['Upgrade to SMB3.1.1', 'Optimize TCP window scaling', 'Configure bypass for migration tools']
                }
            },
            'nonprod_sj_windows_share_fsx_windows': {
                'name': 'Non-Prod: San Jose Windows Share → AWS FSx for Windows',
                'destination_storage': 'FSx_Windows',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Windows',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 3,
                        'reliability': 0.997,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.88
                    },
                    {
                        'name': 'Windows Jump Server to AWS FSx Windows (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 10,
                        'reliability': 0.999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 2.5,
                        'ai_optimization_potential': 0.95
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['SMB protocol optimization', 'Windows AD integration'],
                    'optimization_opportunities': ['Native Windows file system features', 'SMB Direct support', 'Deduplication optimization'],
                    'risk_factors': ['AD integration complexity', 'File system migration'],
                    'recommended_improvements': ['Configure SMB Direct', 'Optimize AD integration', 'Test deduplication settings']
                }
            },
            'nonprod_sj_windows_share_fsx_lustre': {
                'name': 'Non-Prod: San Jose Windows Share → AWS FSx for Lustre',
                'destination_storage': 'FSx_Lustre',
                'source': 'San Jose',
                'destination': 'AWS US-West-2 FSx for Lustre',
                'environment': 'non-production',
                'os_type': 'windows',
                'storage_type': 'share',
                'segments': [
                    {
                        'name': 'Windows Share to Windows Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 3,
                        'reliability': 0.997,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.88
                    },
                    {
                        'name': 'Windows Jump Server to AWS FSx Lustre (DX)',
                        'bandwidth_mbps': 2000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'direct_connect',
                        'cost_factor': 3.5,
                        'ai_optimization_potential': 0.92
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Windows to Lustre compatibility', 'Protocol conversion overhead'],
                    'optimization_opportunities': ['Lustre Windows client optimization', 'Protocol gateway configuration'],
                    'risk_factors': ['Limited Windows Lustre support', 'Complex configuration'],
                    'recommended_improvements': ['Use Lustre gateway', 'Implement protocol translation', 'Test Windows client compatibility']
                }
            },
            'prod_sa_linux_nas_s3': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS Production VPC S3',
                'destination_storage': 'S3',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC S3',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.97
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC S3 (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 8,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.0,
                        'ai_optimization_potential': 0.96
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Cross-site latency accumulation', 'Multiple hop complexity'],
                    'optimization_opportunities': ['End-to-end optimization', 'Compression algorithms', 'Parallel processing'],
                    'risk_factors': ['Multiple failure points', 'Complex troubleshooting', 'Bandwidth contention'],
                    'recommended_improvements': ['Implement WAN optimization', 'Add redundant paths', 'Real-time monitoring']
                }
            },
            'prod_sa_linux_nas_fsx_windows': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS Production VPC FSx for Windows',
                'destination_storage': 'FSx_Windows',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC FSx for Windows',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.97
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC FSx Windows (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 6,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 4.5,
                        'ai_optimization_potential': 0.97
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Linux to Windows protocol conversion', 'Multi-hop latency'],
                    'optimization_opportunities': ['Protocol optimization', 'FSx performance tuning', 'Cross-platform caching'],
                    'risk_factors': ['Multi-site complexity', 'Protocol compatibility'],
                    'recommended_improvements': ['Implement protocol gateways', 'Optimize FSx throughput', 'Add monitoring']
                }
            },
            'prod_sa_linux_nas_fsx_lustre': {
                'name': 'Prod: San Antonio Linux NAS → San Jose → AWS Production VPC FSx for Lustre',
                'destination_storage': 'FSx_Lustre',
                'source': 'San Antonio',
                'destination': 'AWS US-West-2 Production VPC FSx for Lustre',
                'environment': 'production',
                'os_type': 'linux',
                'storage_type': 'nas',
                'segments': [
                    {
                        'name': 'San Antonio Linux NAS to Linux Jump Server',
                        'bandwidth_mbps': 1000,
                        'latency_ms': 1,
                        'reliability': 0.999,
                        'connection_type': 'internal_lan',
                        'cost_factor': 0.0,
                        'ai_optimization_potential': 0.97
                    },
                    {
                        'name': 'San Antonio to San Jose (Private Line)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 12,
                        'reliability': 0.9995,
                        'connection_type': 'private_line',
                        'cost_factor': 3.0,
                        'ai_optimization_potential': 0.94
                    },
                    {
                        'name': 'San Jose to AWS Production VPC FSx Lustre (DX)',
                        'bandwidth_mbps': 10000,
                        'latency_ms': 4,
                        'reliability': 0.9999,
                        'connection_type': 'direct_connect',
                        'cost_factor': 5.0,
                        'ai_optimization_potential': 0.98
                    }
                ],
                'ai_insights': {
                    'performance_bottlenecks': ['Lustre client optimization', 'High-performance requirements'],
                    'optimization_opportunities': ['Lustre striping', 'Parallel I/O optimization', 'Client tuning'],
                    'risk_factors': ['Lustre complexity', 'High-performance requirements'],
                    'recommended_improvements': ['Optimize Lustre configuration', 'Implement parallel processing', 'Monitor performance']
                }
            }
        }
    
    def calculate_ai_enhanced_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """AI-enhanced network path performance calculation"""
        
        path = self.network_paths[path_key]
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        # Original performance calculation (preserved)
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        ai_optimization_score = 1.0
        
        adjusted_segments = []
        
        for segment in path['segments']:
            # Base metrics
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Time-of-day and congestion adjustments (preserved original logic)
            if segment['connection_type'] == 'internal_lan':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.1
                else:
                    congestion_factor = 0.95
            elif segment['connection_type'] == 'private_line':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.2
                else:
                    congestion_factor = 0.9
            elif segment['connection_type'] == 'direct_connect':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.05
                else:
                    congestion_factor = 0.98
            else:
                congestion_factor = 1.0
            
            # Apply congestion
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # OS-specific adjustments (preserved)
            if path['os_type'] == 'windows' and segment['connection_type'] != 'internal_lan':
                effective_bandwidth *= 0.95
                effective_latency *= 1.1
            
            # Destination storage adjustments
            if 'FSx' in path['destination_storage']:
                if path['destination_storage'] == 'FSx_Windows':
                    effective_bandwidth *= 1.1  # Better Windows integration
                    effective_latency *= 0.9    # Lower latency
                elif path['destination_storage'] == 'FSx_Lustre':
                    effective_bandwidth *= 1.3  # High performance
                    effective_latency *= 0.7    # Very low latency
            
            # AI optimization potential
            ai_optimization_score *= segment['ai_optimization_potential']
            
            # Accumulate metrics
            total_latency += effective_latency
            min_bandwidth = min(min_bandwidth, effective_bandwidth)
            total_reliability *= segment_reliability
            total_cost_factor += segment['cost_factor']
            
            adjusted_segments.append({
                **segment,
                'effective_bandwidth_mbps': effective_bandwidth,
                'effective_latency_ms': effective_latency,
                'congestion_factor': congestion_factor
            })
        
        # Calculate quality scores (preserved original logic)
        latency_score = max(0, 100 - (total_latency * 2))
        bandwidth_score = min(100, (min_bandwidth / 1000) * 20)
        reliability_score = total_reliability * 100
        
        # AI-enhanced network quality with optimization potential
        base_network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        ai_enhanced_quality = base_network_quality * ai_optimization_score
        
        # Destination storage performance bonus
        storage_bonus = 0
        if path['destination_storage'] == 'FSx_Windows':
            storage_bonus = 10
        elif path['destination_storage'] == 'FSx_Lustre':
            storage_bonus = 20
        
        ai_enhanced_quality = min(100, ai_enhanced_quality + storage_bonus)
        
        return {
            'path_name': path['name'],
            'destination_storage': path['destination_storage'],
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': base_network_quality,
            'ai_enhanced_quality_score': ai_enhanced_quality,
            'ai_optimization_potential': (1 - ai_optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'storage_performance_bonus': storage_bonus,
            'segments': adjusted_segments,
            'environment': path['environment'],
            'os_type': path['os_type'],
            'storage_type': path['storage_type'],
            'ai_insights': path['ai_insights']
        }
    
    def get_available_paths_by_storage(self, os_type: str, environment: str) -> Dict:
        """Get available network paths grouped by destination storage type"""
        
        storage_groups = {
            'S3': [],
            'FSx_Windows': [],
            'FSx_Lustre': []
        }
        
        for path_key, path_data in self.network_paths.items():
            if (path_data['os_type'] == os_type and 
                path_data['environment'] == environment):
                
                storage_type = path_data['destination_storage']
                if storage_type in storage_groups:
                    storage_groups[storage_type].append({
                        'key': path_key,
                        'name': path_data['name'],
                        'storage_type': storage_type
                    })
        
        return storage_groups

class EnhancedAgentSizingManager:
    """Enhanced agent sizing with scalable agent count and AI recommendations"""
    
    def __init__(self):
        # Enhanced agent configurations with per-agent specifications
        self.datasync_agent_specs = {
            'small': {
                'name': 'Small Agent (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 250,
                'max_concurrent_tasks_per_agent': 10,
                'cost_per_hour_per_agent': 0.0416,
                'recommended_for': 'Up to 1TB per agent, <100 Mbps network per agent',
                'ai_optimization_tips': [
                    'Configure parallel transfers for small files',
                    'Optimize for network bandwidth efficiency',
                    'Consider file compression during transfer'
                ]
            },
            'medium': {
                'name': 'Medium Agent (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 500,
                'max_concurrent_tasks_per_agent': 25,
                'cost_per_hour_per_agent': 0.085,
                'recommended_for': '1-5TB per agent, 100-500 Mbps network per agent',
                'ai_optimization_tips': [
                    'Balance CPU and network utilization',
                    'Implement intelligent retry mechanisms',
                    'Optimize task scheduling algorithms'
                ]
            },
            'large': {
                'name': 'Large Agent (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps_per_agent': 1000,
                'max_concurrent_tasks_per_agent': 50,
                'cost_per_hour_per_agent': 0.17,
                'recommended_for': '5-20TB per agent, 500Mbps-1Gbps network per agent',
                'ai_optimization_tips': [
                    'Enable advanced parallel processing',
                    'Implement adaptive bandwidth management',
                    'Use predictive caching strategies'
                ]
            },
            'xlarge': {
                'name': 'XLarge Agent (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps_per_agent': 2000,
                'max_concurrent_tasks_per_agent': 100,
                'cost_per_hour_per_agent': 0.34,
                'recommended_for': '>20TB per agent, >1Gbps network per agent',
                'ai_optimization_tips': [
                    'Maximize multi-core processing efficiency',
                    'Implement intelligent load balancing',
                    'Use advanced compression algorithms'
                ]
            }
        }
        
        self.dms_agent_specs = {
            'small': {
                'name': 'Small DMS Instance (t3.medium)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 200,
                'max_concurrent_tasks_per_agent': 5,
                'cost_per_hour_per_agent': 0.0416,
                'recommended_for': 'Up to 500GB per agent, simple schemas',
                'ai_optimization_tips': [
                    'Optimize CDC settings for small datasets',
                    'Configure efficient batch sizes',
                    'Minimize transformation overhead'
                ]
            },
            'medium': {
                'name': 'Medium DMS Instance (c5.large)',
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 400,
                'max_concurrent_tasks_per_agent': 10,
                'cost_per_hour_per_agent': 0.085,
                'recommended_for': '500GB-2TB per agent, moderate complexity',
                'ai_optimization_tips': [
                    'Balance memory allocation for transformations',
                    'Implement parallel table loading',
                    'Optimize validation processes'
                ]
            },
            'large': {
                'name': 'Large DMS Instance (c5.xlarge)',
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps_per_agent': 800,
                'max_concurrent_tasks_per_agent': 20,
                'cost_per_hour_per_agent': 0.17,
                'recommended_for': '2-10TB per agent, complex schemas',
                'ai_optimization_tips': [
                    'Enable advanced parallel processing',
                    'Implement intelligent error handling',
                    'Optimize LOB handling strategies'
                ]
            },
            'xlarge': {
                'name': 'XLarge DMS Instance (c5.2xlarge)',
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps_per_agent': 1500,
                'max_concurrent_tasks_per_agent': 40,
                'cost_per_hour_per_agent': 0.34,
                'recommended_for': '10-50TB per agent, very complex schemas',
                'ai_optimization_tips': [
                    'Maximize CPU utilization for transformations',
                    'Implement advanced caching strategies',
                    'Use predictive error prevention'
                ]
            },
            'xxlarge': {
                'name': 'XXLarge DMS Instance (c5.4xlarge)',
                'vcpu': 16,
                'memory_gb': 32,
                'max_throughput_mbps_per_agent': 2500,
                'max_concurrent_tasks_per_agent': 80,
                'cost_per_hour_per_agent': 0.68,
                'recommended_for': '>50TB per agent, enterprise workloads',
                'ai_optimization_tips': [
                    'Optimize for maximum parallel throughput',
                    'Implement intelligent resource allocation',
                    'Use advanced monitoring and auto-tuning'
                ]
            }
        }
    
    def calculate_agent_configuration(self, agent_type: str, agent_size: str, number_of_agents: int, destination_storage: str = 'S3') -> Dict:
        """Calculate total agent configuration based on type, size, count, and destination storage"""
        
        if agent_type == 'datasync':
            agent_spec = self.datasync_agent_specs[agent_size]
        else:  # dms
            agent_spec = self.dms_agent_specs[agent_size]
        
        # Calculate totals with scaling efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(number_of_agents)
        
        # Destination storage performance adjustments
        storage_multiplier = self._get_storage_performance_multiplier(destination_storage)
        
        total_throughput = (agent_spec['max_throughput_mbps_per_agent'] * 
                          number_of_agents * scaling_efficiency * storage_multiplier)
        
        total_concurrent_tasks = (agent_spec['max_concurrent_tasks_per_agent'] * 
                                number_of_agents)
        
        total_cost_per_hour = agent_spec['cost_per_hour_per_agent'] * number_of_agents
        
        # Management overhead calculation
        management_overhead_factor = 1.0 + (number_of_agents - 1) * 0.05  # 5% overhead per additional agent
        
        # Storage-specific overhead
        storage_overhead = self._get_storage_management_overhead(destination_storage)
        
        return {
            'agent_type': agent_type,
            'agent_size': agent_size,
            'number_of_agents': number_of_agents,
            'destination_storage': destination_storage,
            'per_agent_spec': agent_spec,
            'total_vcpu': agent_spec['vcpu'] * number_of_agents,
            'total_memory_gb': agent_spec['memory_gb'] * number_of_agents,
            'max_throughput_mbps_per_agent': agent_spec['max_throughput_mbps_per_agent'],
            'total_max_throughput_mbps': total_throughput,
            'effective_throughput_mbps': total_throughput,  # This will be adjusted by network constraints
            'total_concurrent_tasks': total_concurrent_tasks,
            'cost_per_hour_per_agent': agent_spec['cost_per_hour_per_agent'],
            'total_cost_per_hour': total_cost_per_hour,
            'total_monthly_cost': total_cost_per_hour * 24 * 30,
            'scaling_efficiency': scaling_efficiency,
            'storage_performance_multiplier': storage_multiplier,
            'management_overhead_factor': management_overhead_factor,
            'storage_management_overhead': storage_overhead,
            'effective_cost_per_hour': total_cost_per_hour * management_overhead_factor * storage_overhead,
            'ai_optimization_tips': agent_spec['ai_optimization_tips'],
            'scaling_recommendations': self._get_scaling_recommendations(agent_size, number_of_agents, destination_storage),
            'optimal_configuration': self._assess_configuration_optimality(agent_size, number_of_agents, destination_storage)
        }
    
    def _get_storage_performance_multiplier(self, destination_storage: str) -> float:
        """Get performance multiplier based on destination storage type"""
        multipliers = {
            'S3': 1.0,
            'FSx_Windows': 1.15,
            'FSx_Lustre': 1.4
        }
        return multipliers.get(destination_storage, 1.0)
    
    def _get_storage_management_overhead(self, destination_storage: str) -> float:
        """Get management overhead factor for destination storage"""
        overheads = {
            'S3': 1.0,
            'FSx_Windows': 1.1,
            'FSx_Lustre': 1.2
        }
        return overheads.get(destination_storage, 1.0)
    
    def _calculate_scaling_efficiency(self, number_of_agents: int) -> float:
        """Calculate scaling efficiency - diminishing returns with more agents"""
        if number_of_agents == 1:
            return 1.0
        elif number_of_agents <= 3:
            return 0.95  # 5% overhead for coordination
        elif number_of_agents <= 5:
            return 0.90  # 10% overhead
        elif number_of_agents <= 8:
            return 0.85  # 15% overhead
        else:
            return 0.80  # 20% overhead for complex coordination
    
    def _get_scaling_recommendations(self, agent_size: str, number_of_agents: int, destination_storage: str) -> List[str]:
        """Get scaling-specific recommendations"""
        recommendations = []
        
        if number_of_agents == 1:
            recommendations.append("Single agent configuration - consider scaling for larger workloads")
            recommendations.append("Ensure agent has sufficient resources for peak throughput")
        elif number_of_agents <= 3:
            recommendations.append("Good balance of performance and manageability")
            recommendations.append("Configure load balancing for optimal distribution")
            recommendations.append("Implement agent health monitoring")
        elif number_of_agents <= 5:
            recommendations.append("High-scale configuration requiring careful coordination")
            recommendations.append("Implement centralized monitoring and logging")
            recommendations.append("Consider agent consolidation if underutilized")
        else:
            recommendations.append("Very high-scale configuration - ensure proper orchestration")
            recommendations.append("Implement automated agent management")
            recommendations.append("Monitor for diminishing returns on additional agents")
        
        # Size-specific recommendations
        if agent_size in ['small', 'medium'] and number_of_agents > 5:
            recommendations.append("Consider fewer larger agents instead of many small agents")
        elif agent_size in ['xlarge', 'xxlarge'] and number_of_agents > 3:
            recommendations.append("Large agents may be underutilized - consider workload distribution")
        
        # Storage-specific recommendations
        if destination_storage == 'FSx_Lustre':
            recommendations.append("Optimize agents for high-performance Lustre file system")
            recommendations.append("Configure parallel I/O for FSx Lustre performance")
        elif destination_storage == 'FSx_Windows':
            recommendations.append("Ensure agents are optimized for Windows file sharing protocols")
            recommendations.append("Configure SMB optimization for FSx Windows")
        
        return recommendations
    
    def _assess_configuration_optimality(self, agent_size: str, number_of_agents: int, destination_storage: str) -> Dict:
        """Assess if the configuration is optimal"""
        
        # Scoring system
        efficiency_score = 100
        
        # Penalize for too many small agents
        if agent_size == 'small' and number_of_agents > 6:
            efficiency_score -= 20
        
        # Penalize for too few large agents when they could be better utilized
        if agent_size in ['xlarge', 'xxlarge'] and number_of_agents == 1:
            efficiency_score -= 10
        
        # Penalize for management complexity
        if number_of_agents > 8:
            efficiency_score -= 25
        
        # Optimal ranges
        if 2 <= number_of_agents <= 4 and agent_size in ['medium', 'large']:
            efficiency_score += 10
        
        # Storage-specific adjustments
        if destination_storage == 'FSx_Lustre' and agent_size in ['large', 'xlarge']:
            efficiency_score += 5  # Bonus for high-performance storage
        elif destination_storage == 'FSx_Windows' and agent_size in ['medium', 'large']:
            efficiency_score += 3  # Bonus for balanced Windows performance
        
        # Management complexity assessment
        if number_of_agents <= 2:
            complexity = "Low"
        elif number_of_agents <= 5:
            complexity = "Medium"
        else:
            complexity = "High"
        
        # Cost efficiency
        cost_efficiency = "Good" if efficiency_score >= 90 else "Fair" if efficiency_score >= 75 else "Poor"
        
        return {
            'efficiency_score': max(0, efficiency_score),
            'management_complexity': complexity,
            'cost_efficiency': cost_efficiency,
            'storage_optimization': self._get_storage_optimization_rating(destination_storage, agent_size),
            'optimal_recommendation': self._generate_optimal_recommendation(agent_size, number_of_agents, efficiency_score, destination_storage)
        }
    
    def _get_storage_optimization_rating(self, destination_storage: str, agent_size: str) -> str:
        """Get storage optimization rating"""
        if destination_storage == 'FSx_Lustre':
            return "Excellent" if agent_size in ['large', 'xlarge'] else "Good"
        elif destination_storage == 'FSx_Windows':
            return "Very Good" if agent_size in ['medium', 'large'] else "Good"
        else:  # S3
            return "Good"
    
    def _generate_optimal_recommendation(self, agent_size: str, number_of_agents: int, efficiency_score: int, destination_storage: str) -> str:
        """Generate optimal configuration recommendation"""
        
        if efficiency_score >= 90:
            return f"Optimal configuration for {destination_storage} destination"
        elif efficiency_score >= 75:
            return f"Good configuration with minor optimization opportunities for {destination_storage}"
        elif number_of_agents > 6 and agent_size == 'small':
            return "Consider consolidating to fewer, larger agents"
        elif number_of_agents > 8:
            return "Consider reducing agent count to improve manageability"
        elif agent_size in ['xlarge', 'xxlarge'] and number_of_agents == 1:
            return "Consider multiple medium/large agents for better distribution"
        elif destination_storage == 'FSx_Lustre' and agent_size == 'small':
            return "Consider larger agents to fully utilize FSx Lustre performance"
        else:
            return f"Configuration needs optimization for better {destination_storage} efficiency"
    
    def recommend_optimal_agents(self, database_size_gb: int, network_bandwidth_mbps: int, 
                                migration_window_hours: int, destination_storage: str = 'S3') -> Dict:
        """AI-powered recommendation for optimal agent configuration including storage destination"""
        
        # Calculate required throughput
        required_throughput_mbps = (database_size_gb * 8 * 1000) / (migration_window_hours * 3600)
        
        # Consider network constraints
        effective_required_throughput = min(required_throughput_mbps, network_bandwidth_mbps * 0.8)
        
        # Adjust for destination storage performance
        storage_multiplier = self._get_storage_performance_multiplier(destination_storage)
        storage_adjusted_throughput = effective_required_throughput / storage_multiplier
        
        recommendations = []
        
        # Evaluate different configurations
        for agent_type in ['datasync', 'dms']:
            agent_specs = self.datasync_agent_specs if agent_type == 'datasync' else self.dms_agent_specs
            
            for size in agent_specs.keys():
                for num_agents in range(1, 9):  # Test 1-8 agents
                    config = self.calculate_agent_configuration(agent_type, size, num_agents, destination_storage)
                    
                    if config['total_max_throughput_mbps'] >= storage_adjusted_throughput:
                        # Calculate efficiency score
                        throughput_ratio = storage_adjusted_throughput / config['total_max_throughput_mbps']
                        cost_efficiency = 1 / config['total_cost_per_hour']
                        scaling_penalty = 1 - ((num_agents - 1) * 0.1)  # Penalty for complexity
                        storage_bonus = storage_multiplier - 1  # Bonus for high-performance storage
                        
                        overall_score = (throughput_ratio * 0.4 + 
                                       cost_efficiency * 100 * 0.3 + 
                                       scaling_penalty * 0.2 +
                                       storage_bonus * 0.1) * 100
                        
                        recommendations.append({
                            'agent_type': agent_type,
                            'agent_size': size,
                            'number_of_agents': num_agents,
                            'destination_storage': destination_storage,
                            'total_throughput_mbps': config['total_max_throughput_mbps'],
                            'total_cost_per_hour': config['total_cost_per_hour'],
                            'overall_score': overall_score,
                            'storage_performance_multiplier': storage_multiplier,
                            'configuration': config
                        })
        
        # Sort by overall score
        recommendations.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            'required_throughput_mbps': effective_required_throughput,
            'storage_adjusted_throughput_mbps': storage_adjusted_throughput,
            'destination_storage': destination_storage,
            'storage_performance_multiplier': storage_multiplier,
            'top_recommendations': recommendations[:5],
            'optimal_configuration': recommendations[0] if recommendations else None,
            'analysis_parameters': {
                'database_size_gb': database_size_gb,
                'network_bandwidth_mbps': network_bandwidth_mbps,
                'migration_window_hours': migration_window_hours,
                'destination_storage': destination_storage
            }
        }

class EnhancedAWSMigrationManager:
    """Enhanced AWS migration manager with AI and real-time pricing"""
    
    def __init__(self, aws_api_manager: AWSAPIManager, ai_manager: AnthropicAIManager):
        self.aws_api = aws_api_manager
        self.ai_manager = ai_manager
        
        # Keep all original migration types
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
    
    async def ai_enhanced_aws_sizing(self, on_prem_config: Dict) -> Dict:
        """AI-enhanced AWS sizing with real-time pricing"""
        
        # Get real-time pricing
        pricing_data = await self.aws_api.get_real_time_pricing()
        
        # Get AI workload analysis
        ai_analysis = await self.ai_manager.analyze_migration_workload(
            on_prem_config, 
            on_prem_config.get('performance_data', {})
        )
        
        # Enhanced sizing logic with AI insights
        rds_recommendations = await self._ai_calculate_rds_sizing(
            on_prem_config, pricing_data, ai_analysis
        )
        
        ec2_recommendations = await self._ai_calculate_ec2_sizing(
            on_prem_config, pricing_data, ai_analysis
        )
        
        # AI-enhanced reader/writer configuration
        reader_writer_config = await self._ai_calculate_reader_writer_config(
            on_prem_config, ai_analysis
        )
        
        # AI-powered deployment recommendation
        deployment_recommendation = await self._ai_recommend_deployment_type(
            on_prem_config, ai_analysis, rds_recommendations, ec2_recommendations
        )
        
        return {
            'rds_recommendations': rds_recommendations,
            'ec2_recommendations': ec2_recommendations,
            'reader_writer_config': reader_writer_config,
            'deployment_recommendation': deployment_recommendation,
            'ai_analysis': ai_analysis,
            'pricing_data': pricing_data
        }
    
    async def _ai_calculate_rds_sizing(self, config: Dict, pricing_data: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced RDS sizing calculation"""
        
        # Extract configuration
        cpu_cores = config['cpu_cores']
        ram_gb = config['ram_gb']
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        
        # AI-based sizing multipliers
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.1  # Adjust based on AI complexity
        
        # Agent scaling impact on sizing
        num_agents = config.get('number_of_agents', 1)
        agent_scaling_factor = 1.0 + (num_agents - 1) * 0.05  # More agents may need more DB resources
        
        cpu_multiplier = (1.2 if performance_req == 'high' else 1.0) * complexity_multiplier * agent_scaling_factor
        memory_multiplier = (1.3 if database_engine in ['oracle', 'postgresql'] else 1.1) * complexity_multiplier
        
        # Required cloud resources with AI adjustment
        required_vcpu = max(2, int(cpu_cores * cpu_multiplier))
        required_memory = max(8, int(ram_gb * memory_multiplier))
        
        # Use real-time pricing data
        rds_instances = pricing_data.get('rds_instances', {})
        
        # AI-enhanced instance selection
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in rds_instances.items():
            if specs['vcpu'] >= required_vcpu and specs['memory'] >= required_memory:
                # AI-enhanced scoring
                cpu_waste = specs['vcpu'] - required_vcpu
                memory_waste = specs['memory'] - required_memory
                cost_factor = specs['cost_per_hour']
                
                # AI complexity penalty for oversizing
                ai_penalty = 0 if ai_complexity <= 6 else (ai_complexity - 6) * 0.1
                
                score = (cpu_waste * 0.3 + memory_waste * 0.001 + cost_factor * 0.5 + ai_penalty)
                
                if score < best_score:
                    best_score = score
                    best_instance = instance_type
        
        if not best_instance:
            best_instance = 'db.r6g.8xlarge'
        
        # AI-enhanced storage recommendations
        storage_multiplier = 1.5 + (ai_complexity - 5) * 0.1  # More storage for complex migrations
        storage_size_gb = max(database_size_gb * storage_multiplier, 100)
        storage_type = 'io1' if database_size_gb > 5000 or performance_req == 'high' or ai_complexity > 7 else 'gp3'
        
        # Calculate costs with real-time pricing
        instance_cost = rds_instances[best_instance]['cost_per_hour'] * 24 * 30
        storage_cost = storage_size_gb * pricing_data.get('storage', {}).get(storage_type, {}).get('cost_per_gb_month', 0.08)
        
        return {
            'primary_instance': best_instance,
            'instance_specs': rds_instances[best_instance],
            'storage_type': storage_type,
            'storage_size_gb': storage_size_gb,
            'monthly_instance_cost': instance_cost,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': instance_cost + storage_cost,
            'multi_az': environment == 'production',
            'backup_retention_days': 30 if environment == 'production' else 7,
            'ai_sizing_factors': {
                'complexity_multiplier': complexity_multiplier,
                'agent_scaling_factor': agent_scaling_factor,
                'ai_complexity_score': ai_complexity,
                'storage_multiplier': storage_multiplier
            },
            'ai_recommendations': ai_analysis.get('performance_recommendations', [])
        }
    
    async def _ai_calculate_ec2_sizing(self, config: Dict, pricing_data: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced EC2 sizing calculation"""
        
        # Similar to RDS but with EC2-specific adjustments
        cpu_cores = config['cpu_cores']
        ram_gb = config['ram_gb']
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        
        # AI complexity-based adjustments
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.15  # More aggressive for EC2
        
        # Agent scaling impact
        num_agents = config.get('number_of_agents', 1)
        agent_scaling_factor = 1.0 + (num_agents - 1) * 0.08  # More EC2 resources for multi-agent coordination
        
        # EC2 needs more overhead for OS and database management
        cpu_multiplier = (1.4 if performance_req == 'high' else 1.2) * complexity_multiplier * agent_scaling_factor
        memory_multiplier = (1.5 if database_engine in ['oracle', 'postgresql'] else 1.3) * complexity_multiplier
        
        required_vcpu = max(2, int(cpu_cores * cpu_multiplier))
        required_memory = max(8, int(ram_gb * memory_multiplier))
        
        # Use real-time pricing
        ec2_instances = pricing_data.get('ec2_instances', {})
        
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in ec2_instances.items():
            if specs['vcpu'] >= required_vcpu and specs['memory'] >= required_memory:
                cpu_waste = specs['vcpu'] - required_vcpu
                memory_waste = specs['memory'] - required_memory
                cost_factor = specs['cost_per_hour']
                
                # AI penalty for complex workloads
                ai_penalty = 0 if ai_complexity <= 6 else (ai_complexity - 6) * 0.15
                
                score = (cpu_waste * 0.3 + memory_waste * 0.001 + cost_factor * 0.5 + ai_penalty)
                
                if score < best_score:
                    best_score = score
                    best_instance = instance_type
        
        if not best_instance:
            best_instance = 'r6i.8xlarge'
        
        # AI-enhanced storage sizing for EC2
        storage_multiplier = 2.0 + (ai_complexity - 5) * 0.2  # Even more generous for EC2
        storage_size_gb = max(database_size_gb * storage_multiplier, 100)
        storage_type = 'io2' if performance_req == 'high' or ai_complexity > 7 else 'gp3'
        
        # Calculate costs
        instance_cost = ec2_instances[best_instance]['cost_per_hour'] * 24 * 30
        storage_cost = storage_size_gb * pricing_data.get('storage', {}).get(storage_type, {}).get('cost_per_gb_month', 0.08)
        
        return {
            'primary_instance': best_instance,
            'instance_specs': ec2_instances[best_instance],
            'storage_type': storage_type,
            'storage_size_gb': storage_size_gb,
            'monthly_instance_cost': instance_cost,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': instance_cost + storage_cost,
            'ebs_optimized': True,
            'enhanced_networking': True,
            'ai_sizing_factors': {
                'complexity_multiplier': complexity_multiplier,
                'agent_scaling_factor': agent_scaling_factor,
                'ai_complexity_score': ai_complexity,
                'storage_multiplier': storage_multiplier
            },
            'ai_recommendations': ai_analysis.get('performance_recommendations', [])
        }
    
    async def _ai_calculate_reader_writer_config(self, config: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced reader/writer configuration"""
        
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        environment = config.get('environment', 'non-production')
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        num_agents = config.get('number_of_agents', 1)
        
        # Start with single writer
        writers = 1
        readers = 0
        
        # AI-enhanced scaling logic
        if database_size_gb > 1000:
            readers += 1
        if database_size_gb > 5000:
            readers += 1 + int(ai_complexity / 5)  # AI complexity adds more readers
        if database_size_gb > 20000:
            readers += 2 + int(ai_complexity / 3)
        
        # Performance-based scaling with AI insights
        if performance_req == 'high':
            readers += 2 + int(ai_complexity / 4)
        
        # Agent scaling impact - more agents may benefit from more read replicas
        if num_agents > 2:
            readers += 1 + (num_agents // 3)
        
        # Environment-based scaling
        if environment == 'production':
            readers = max(readers, 2)  # Minimum 2 readers for production
            if database_size_gb > 50000 or ai_complexity > 8:
                writers = 2  # Multi-writer for very large or complex production DBs
        
        # AI-recommended adjustments
        if 'high availability' in ' '.join(ai_analysis.get('best_practices', [])).lower():
            readers += 1
        if 'complex queries' in ' '.join(ai_analysis.get('risk_factors', [])).lower():
            readers += 1
        
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
            'recommended_read_split': min(80, read_capacity_percent),
            'reasoning': f"AI-optimized for {database_size_gb}GB, complexity {ai_complexity}/10, {performance_req} performance, {environment}, {num_agents} agents",
            'ai_insights': {
                'complexity_impact': ai_complexity,
                'agent_scaling_impact': num_agents,
                'scaling_factors': ai_analysis.get('performance_recommendations', [])[:3],
                'optimization_potential': f"{(10 - ai_complexity) * 10}% potential for further optimization"
            }
        }
    
    async def _ai_recommend_deployment_type(self, config: Dict, ai_analysis: Dict, 
                                          rds_rec: Dict, ec2_rec: Dict) -> Dict:
        """AI-powered deployment type recommendation"""
        
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        num_agents = config.get('number_of_agents', 1)
        
        rds_score = 0
        ec2_score = 0
        
        # Size-based scoring (preserved original logic)
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
            ec2_score += 25
        
        # Environment scoring
        if environment == 'production':
            rds_score += 20
        else:
            ec2_score += 10
        
        # AI complexity scoring
        if ai_complexity > 7:
            ec2_score += 20  # Complex workloads might need more control
            rds_score += 5   # But managed services help with complexity
        elif ai_complexity < 4:
            rds_score += 25  # Simple workloads perfect for managed services
        
        # Agent scaling considerations
        if num_agents > 3:
            ec2_score += 15  # Multi-agent setups may need more control
        elif num_agents == 1:
            rds_score += 10  # Single agent works well with managed services
        
        # AI insights-based scoring
        risk_factors = ai_analysis.get('risk_factors', [])
        if any('performance' in risk.lower() for risk in risk_factors):
            ec2_score += 15
        if any('management' in risk.lower() or 'operational' in risk.lower() for risk in risk_factors):
            rds_score += 20
        
        # Management complexity
        rds_score += 20
        
        # Cost consideration
        rds_cost = rds_rec.get('total_monthly_cost', 0)
        ec2_cost = ec2_rec.get('total_monthly_cost', 0)
        
        if ec2_cost < rds_cost * 0.8:  # EC2 significantly cheaper
            ec2_score += 15
        elif rds_cost < ec2_cost * 0.9:  # RDS competitive
            rds_score += 10
        
        recommendation = 'rds' if rds_score > ec2_score else 'ec2'
        confidence = abs(rds_score - ec2_score) / max(rds_score, ec2_score, 1)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rds_score': rds_score,
            'ec2_score': ec2_score,
            'ai_complexity_factor': ai_complexity,
            'agent_scaling_factor': num_agents,
            'primary_reasons': self._get_ai_deployment_reasons(
                recommendation, rds_score, ec2_score, ai_analysis, num_agents
            ),
            'ai_insights': {
                'complexity_impact': f"AI complexity score {ai_complexity}/10 influenced recommendation",
                'agent_impact': f"{num_agents} agents affected scoring",
                'risk_mitigation': ai_analysis.get('mitigation_strategies', [])[:3],
                'cost_factors': {
                    'rds_monthly': rds_cost,
                    'ec2_monthly': ec2_cost,
                    'cost_difference_percent': abs(rds_cost - ec2_cost) / max(rds_cost, ec2_cost, 1) * 100
                }
            }
        }
    
    def _get_ai_deployment_reasons(self, recommendation: str, rds_score: int, 
                                 ec2_score: int, ai_analysis: Dict, num_agents: int) -> List[str]:
        """Get AI-enhanced reasons for deployment recommendation"""
        
        base_reasons = []
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        
        if recommendation == 'rds':
            base_reasons = [
                "AI-optimized managed service reduces operational complexity",
                "Automated backups and patching aligned with AI recommendations",
                "Built-in monitoring supports AI-driven optimization",
                "Easy scaling matches AI-predicted growth patterns",
                f"AI complexity score ({ai_complexity}/10) suggests managed service benefits"
            ]
            
            if num_agents <= 2:
                base_reasons.append(f"Simple {num_agents}-agent setup works well with managed RDS")
        else:
            base_reasons = [
                "Maximum control needed for AI-identified performance requirements",
                "Complex configurations support AI optimization strategies",
                "Custom tuning capabilities for AI-recommended improvements",
                "Full control enables AI-driven performance optimization",
                f"AI complexity score ({ai_complexity}/10) suggests need for advanced control"
            ]
            
            if num_agents > 3:
                base_reasons.append(f"{num_agents}-agent coordination benefits from EC2 flexibility")
        
        # Add AI-specific insights
        ai_recommendations = ai_analysis.get('performance_recommendations', [])
        if ai_recommendations:
            base_reasons.append(f"Supports AI recommendation: {ai_recommendations[0][:50]}...")
        
        return base_reasons[:6]

class OnPremPerformanceAnalyzer:
    """Enhanced on-premises performance analyzer with AI insights"""
    
    def __init__(self):
        # Keep all original CPU, storage configurations
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
    
    def calculate_ai_enhanced_performance(self, config: Dict, os_manager: OSPerformanceManager) -> Dict:
        """AI-enhanced on-premises performance calculation"""
        
        # Get original OS impact
        os_impact = os_manager.calculate_os_performance_impact(
            config['operating_system'], 
            config['server_type'], 
            config['database_engine']
        )
        
        # Original performance calculations (preserved)
        cpu_performance = self._calculate_cpu_performance(config, os_impact)
        memory_performance = self._calculate_memory_performance(config, os_impact)
        storage_performance = self._calculate_storage_performance(config, os_impact)
        network_performance = self._calculate_network_performance(config, os_impact)
        database_performance = self._calculate_database_performance(config, os_impact)
        
        # Enhanced performance analysis using new metrics
        current_performance_analysis = self._analyze_current_performance_metrics(config)
        
        # AI-enhanced overall performance analysis
        overall_performance = self._calculate_ai_enhanced_overall_performance(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, database_performance, os_impact, config
        )
        
        # AI bottleneck analysis
        ai_bottlenecks = self._ai_identify_bottlenecks(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, config
        )
        
        # Resource adequacy analysis
        resource_adequacy = self._analyze_resource_adequacy(config)
        
        return {
            'cpu_performance': cpu_performance,
            'memory_performance': memory_performance,
            'storage_performance': storage_performance,
            'network_performance': network_performance,
            'database_performance': database_performance,
            'current_performance_analysis': current_performance_analysis,
            'resource_adequacy': resource_adequacy,
            'overall_performance': overall_performance,
            'os_impact': os_impact,
            'bottlenecks': ai_bottlenecks['bottlenecks'],
            'ai_insights': ai_bottlenecks['ai_insights'],
            'performance_score': overall_performance['composite_score'],
            'ai_optimization_recommendations': self._generate_ai_optimization_recommendations(
                overall_performance, os_impact, config
            )
        }
    
    def _analyze_current_performance_metrics(self, config: Dict) -> Dict:
        """Analyze current performance metrics provided by user"""
        
        current_storage = config.get('current_storage_gb', 0)
        peak_iops = config.get('peak_iops', 0)
        max_throughput = config.get('max_throughput_mbps', 0)
        database_size = config.get('database_size_gb', 0)
        
        # Storage utilization analysis
        storage_utilization = (database_size / current_storage) * 100 if current_storage > 0 else 0
        
        # IOPS intensity analysis
        iops_per_gb = peak_iops / database_size if database_size > 0 else 0
        
        # Throughput efficiency
        throughput_per_gb = max_throughput / database_size if database_size > 0 else 0
        
        # Performance classification
        if iops_per_gb > 50:
            workload_type = "High IOPS (OLTP-intensive)"
        elif throughput_per_gb > 1:
            workload_type = "High Throughput (Analytics/Batch)"
        else:
            workload_type = "Balanced Workload"
        
        return {
            'storage_utilization_percent': storage_utilization,
            'iops_per_gb': iops_per_gb,
            'throughput_per_gb_mbps': throughput_per_gb,
            'workload_classification': workload_type,
            'storage_efficiency': min(100, (100 - storage_utilization) + 50),  # Higher efficiency for lower utilization
            'performance_intensity': min(100, (iops_per_gb * 2) + (throughput_per_gb * 10)),
            'optimization_priority': "High" if storage_utilization > 80 or iops_per_gb > 100 else "Medium" if storage_utilization > 60 else "Low"
        }
    
    def _analyze_resource_adequacy(self, config: Dict) -> Dict:
        """Analyze resource adequacy comparing current vs anticipated needs"""
        
        current_memory = config.get('ram_gb', 0)
        anticipated_memory = config.get('anticipated_max_memory_gb', 0)
        current_cpu = config.get('cpu_cores', 0)
        anticipated_cpu = config.get('anticipated_max_cpu_cores', 0)
        
        # Memory adequacy
        memory_gap = anticipated_memory - current_memory
        memory_adequacy_score = (current_memory / anticipated_memory) * 100 if anticipated_memory > 0 else 100
        
        # CPU adequacy
        cpu_gap = anticipated_cpu - current_cpu
        cpu_adequacy_score = (current_cpu / anticipated_cpu) * 100 if anticipated_cpu > 0 else 100
        
        # Overall readiness
        overall_adequacy = (memory_adequacy_score + cpu_adequacy_score) / 2
        
        # Readiness classification
        if overall_adequacy >= 90:
            readiness_level = "Excellent - Current resources meet anticipated needs"
        elif overall_adequacy >= 75:
            readiness_level = "Good - Minor upgrades may be beneficial"
        elif overall_adequacy >= 60:
            readiness_level = "Fair - Moderate upgrades recommended"
        else:
            readiness_level = "Poor - Significant upgrades required"
        
        return {
            'memory_gap_gb': memory_gap,
            'cpu_gap_cores': cpu_gap,
            'memory_adequacy_score': memory_adequacy_score,
            'cpu_adequacy_score': cpu_adequacy_score,
            'overall_adequacy_score': overall_adequacy,
            'readiness_level': readiness_level,
            'upgrade_priority': "Immediate" if overall_adequacy < 60 else "Short-term" if overall_adequacy < 75 else "Long-term" if overall_adequacy < 90 else "Optional"
        }
    
    # Keep all original calculation methods but add AI enhancements
    def _calculate_cpu_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate CPU performance metrics with AI insights"""
        
        # Original calculation (preserved)
        base_performance = config['cpu_cores'] * config['cpu_ghz']
        os_adjusted = base_performance * os_impact['cpu_efficiency']
        
        if config['server_type'] == 'vmware':
            virtualization_penalty = 1 - os_impact['virtualization_overhead']
            final_performance = os_adjusted * virtualization_penalty
        else:
            final_performance = os_adjusted * 1.05
        
        single_thread_perf = config['cpu_ghz'] * os_impact['cpu_efficiency']
        multi_thread_perf = final_performance
        
        # AI enhancement: predict scaling characteristics
        ai_scaling_prediction = self._predict_cpu_scaling(config, final_performance)
        
        return {
            'base_performance': base_performance,
            'os_adjusted_performance': os_adjusted,
            'final_performance': final_performance,
            'single_thread_performance': single_thread_perf,
            'multi_thread_performance': multi_thread_perf,
            'utilization_estimate': 0.7,
            'efficiency_factor': os_impact['cpu_efficiency'],
            'ai_scaling_prediction': ai_scaling_prediction,
            'ai_bottleneck_risk': 'high' if final_performance < 30 else 'medium' if final_performance < 60 else 'low'
        }
    
    def _calculate_memory_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate memory performance with AI insights"""
        
        # Original calculation (preserved)
        base_memory = config['ram_gb']
        
        if 'windows' in config['operating_system']:
            os_overhead = 4
        else:
            os_overhead = 2
        
        available_memory = base_memory - os_overhead
        db_memory = available_memory * 0.8
        buffer_pool = db_memory * 0.7
        memory_efficiency = os_impact['memory_efficiency']
        effective_memory = available_memory * memory_efficiency
        
        # AI enhancement: memory pressure prediction
        ai_memory_analysis = self._analyze_memory_requirements(config, effective_memory)
        
        return {
            'total_memory_gb': base_memory,
            'os_overhead_gb': os_overhead,
            'available_memory_gb': available_memory,
            'database_memory_gb': db_memory,
            'buffer_pool_gb': buffer_pool,
            'effective_memory_gb': effective_memory,
            'memory_efficiency': memory_efficiency,
            'memory_pressure': 'low' if available_memory > 32 else 'medium' if available_memory > 16 else 'high',
            'ai_memory_analysis': ai_memory_analysis,
            'ai_optimization_potential': ai_memory_analysis['optimization_potential']
        }
    
    def _calculate_storage_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate storage performance with AI insights"""
        
        # Original calculation (preserved)
        if config['cpu_cores'] >= 8:
            storage_type = 'nvme_ssd'
        elif config['cpu_cores'] >= 4:
            storage_type = 'sata_ssd'
        else:
            storage_type = 'sas_hdd'
        
        storage_specs = self.storage_types[storage_type]
        
        effective_iops = storage_specs['iops'] * os_impact['io_efficiency']
        effective_throughput = storage_specs['throughput_mbps'] * os_impact['io_efficiency']
        effective_latency = storage_specs['latency_ms'] / os_impact['io_efficiency']
        
        # AI enhancement: storage optimization analysis
        ai_storage_analysis = self._analyze_storage_optimization(config, storage_type, effective_iops)
        
        return {
            'storage_type': storage_type,
            'base_iops': storage_specs['iops'],
            'effective_iops': effective_iops,
            'base_throughput_mbps': storage_specs['throughput_mbps'],
            'effective_throughput_mbps': effective_throughput,
            'base_latency_ms': storage_specs['latency_ms'],
            'effective_latency_ms': effective_latency,
            'io_efficiency': os_impact['io_efficiency'],
            'ai_storage_analysis': ai_storage_analysis,
            'ai_upgrade_recommendation': ai_storage_analysis['upgrade_recommendation']
        }
    
    def _calculate_network_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate network performance with AI insights"""
        
        # Original calculation (preserved)
        base_bandwidth = config['nic_speed']
        effective_bandwidth = base_bandwidth * os_impact['network_efficiency']
        
        if config['server_type'] == 'vmware':
            effective_bandwidth *= 0.92
        
        # AI enhancement: network optimization analysis
        ai_network_analysis = self._analyze_network_optimization(config, effective_bandwidth)
        
        return {
            'nic_type': config['nic_type'],
            'base_bandwidth_mbps': base_bandwidth,
            'effective_bandwidth_mbps': effective_bandwidth,
            'network_efficiency': os_impact['network_efficiency'],
            'estimated_latency_ms': 0.1 if 'fiber' in config['nic_type'] else 0.2,
            'ai_network_analysis': ai_network_analysis,
            'ai_optimization_score': ai_network_analysis['optimization_score']
        }
    
    def _calculate_database_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate database performance with AI insights"""
        
        # Original calculation (preserved)
        db_optimization = os_impact['db_optimization']
        
        if config['database_engine'] == 'mysql':
            base_tps = 5000
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
        else:
            base_tps = 4000
            connection_limit = 800
        
        hardware_factor = min(2.0, (config['cpu_cores'] / 4) * (config['ram_gb'] / 16))
        effective_tps = base_tps * hardware_factor * db_optimization
        
        # AI enhancement: database optimization analysis
        ai_db_analysis = self._analyze_database_optimization(config, effective_tps, db_optimization)
        
        return {
            'database_engine': config['database_engine'],
            'base_tps': base_tps,
            'hardware_factor': hardware_factor,
            'db_optimization': db_optimization,
            'effective_tps': effective_tps,
            'max_connections': connection_limit,
            'query_cache_efficiency': db_optimization * 0.9,
            'ai_db_analysis': ai_db_analysis,
            'ai_tuning_potential': ai_db_analysis['tuning_potential']
        }
    
    def _calculate_ai_enhanced_overall_performance(self, cpu_perf: Dict, mem_perf: Dict, 
                                                 storage_perf: Dict, net_perf: Dict, 
                                                 db_perf: Dict, os_impact: Dict, config: Dict) -> Dict:
        """AI-enhanced overall performance calculation"""
        
        # Original scoring (preserved)
        cpu_score = min(100, (cpu_perf['final_performance'] / 50) * 100)
        memory_score = min(100, (mem_perf['effective_memory_gb'] / 64) * 100)
        storage_score = min(100, (storage_perf['effective_iops'] / 100000) * 100)
        network_score = min(100, (net_perf['effective_bandwidth_mbps'] / 10000) * 100)
        database_score = min(100, (db_perf['effective_tps'] / 10000) * 100)
        
        # Original composite score
        composite_score = (
            cpu_score * 0.25 +
            memory_score * 0.2 +
            storage_score * 0.25 +
            network_score * 0.15 +
            database_score * 0.15
        )
        
        # AI enhancement: workload-specific optimization
        ai_workload_optimization = self._analyze_workload_optimization(config, {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score
        })
        
        # AI-adjusted composite score
        ai_adjusted_score = composite_score * ai_workload_optimization['optimization_factor']
        
        return {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score,
            'composite_score': composite_score,
            'ai_adjusted_score': ai_adjusted_score,
            'performance_tier': self._get_performance_tier(ai_adjusted_score),
            'scaling_recommendation': self._get_ai_scaling_recommendation(cpu_score, memory_score, storage_score),
            'ai_workload_optimization': ai_workload_optimization
        }
    
    # AI helper methods
    def _predict_cpu_scaling(self, config: Dict, performance: float) -> Dict:
        """AI-powered CPU scaling prediction"""
        database_size = config['database_size_gb']
        
        scaling_factor = 1.0
        if database_size > 10000:
            scaling_factor = 1.3
        elif database_size > 5000:
            scaling_factor = 1.15
        
        return {
            'predicted_scaling_needs': scaling_factor,
            'bottleneck_prediction': 'likely' if performance < 40 else 'possible' if performance < 70 else 'unlikely',
            'recommendation': 'Upgrade CPU' if performance < 40 else 'Monitor performance' if performance < 70 else 'Current CPU sufficient'
        }
    
    def _analyze_memory_requirements(self, config: Dict, effective_memory: float) -> Dict:
        """AI memory requirement analysis"""
        database_size = config['database_size_gb']
        
        recommended_memory = database_size * 0.1  # 10% of database size as baseline
        if config['database_engine'] in ['oracle', 'postgresql']:
            recommended_memory *= 1.5
        
        optimization_potential = max(0, (recommended_memory - effective_memory) / recommended_memory)
        
        return {
            'recommended_memory_gb': recommended_memory,
            'current_vs_recommended': effective_memory / recommended_memory,
            'optimization_potential': optimization_potential,
            'memory_adequacy': 'sufficient' if effective_memory >= recommended_memory else 'insufficient'
        }
    
    def _analyze_storage_optimization(self, config: Dict, storage_type: str, effective_iops: float) -> Dict:
        """AI storage optimization analysis"""
        database_size = config['database_size_gb']
        
        # Predict IOPS requirements based on database size and type
        if config['database_engine'] in ['oracle', 'sqlserver']:
            required_iops = database_size * 5  # Higher IOPS requirement
        else:
            required_iops = database_size * 3
        
        upgrade_needed = effective_iops < required_iops
        
        return {
            'required_iops': required_iops,
            'current_vs_required': effective_iops / required_iops,
            'upgrade_recommendation': 'Upgrade to NVMe SSD' if upgrade_needed and storage_type != 'nvme_ssd' else 'Current storage adequate',
            'performance_impact': 'high' if upgrade_needed else 'low'
        }
    
    def _analyze_network_optimization(self, config: Dict, effective_bandwidth: float) -> Dict:
        """AI network optimization analysis"""
        database_size = config['database_size_gb']
        
        # Predict network requirements
        required_bandwidth = min(10000, database_size * 10)  # 10 Mbps per GB, max 10 Gbps
        
        optimization_score = min(100, (effective_bandwidth / required_bandwidth) * 100)
        
        return {
            'required_bandwidth_mbps': required_bandwidth,
            'optimization_score': optimization_score,
            'bottleneck_risk': 'high' if optimization_score < 50 else 'medium' if optimization_score < 80 else 'low'
        }
    
    def _analyze_database_optimization(self, config: Dict, effective_tps: float, db_optimization: float) -> Dict:
        """AI database optimization analysis"""
        
        # Predict TPS requirements based on database characteristics
        base_requirement = 2000  # Base TPS requirement
        if config['performance_requirements'] == 'high':
            base_requirement *= 2
        if config['database_size_gb'] > 10000:
            base_requirement *= 1.5
        
        tuning_potential = (1 - db_optimization) * 100
        
        return {
            'required_tps': base_requirement,
            'current_vs_required': effective_tps / base_requirement,
            'tuning_potential': tuning_potential,
            'optimization_priority': 'high' if tuning_potential > 20 else 'medium' if tuning_potential > 10 else 'low'
        }
    
    def _analyze_workload_optimization(self, config: Dict, scores: Dict) -> Dict:
        """AI workload-specific optimization analysis"""
        
        # Identify workload pattern
        if config['database_engine'] in ['mysql', 'postgresql']:
            workload_type = 'oltp'
            cpu_weight = 0.3
            memory_weight = 0.25
            storage_weight = 0.3
        elif config['database_engine'] == 'oracle':
            workload_type = 'mixed'
            cpu_weight = 0.25
            memory_weight = 0.3
            storage_weight = 0.25
        else:
            workload_type = 'general'
            cpu_weight = 0.25
            memory_weight = 0.2
            storage_weight = 0.25
        
        # Calculate workload-specific optimization factor
        weighted_score = (
            scores['cpu_score'] * cpu_weight +
            scores['memory_score'] * memory_weight +
            scores['storage_score'] * storage_weight +
            scores['network_score'] * 0.1 +
            scores['database_score'] * 0.1
        )
        
        optimization_factor = weighted_score / 100
        
        return {
            'workload_type': workload_type,
            'optimization_factor': optimization_factor,
            'primary_bottleneck': max(scores.items(), key=lambda x: x[1])[0].replace('_score', ''),
            'recommended_focus': self._get_optimization_focus(scores)
        }
    
    def _get_optimization_focus(self, scores: Dict) -> str:
        """Get primary optimization focus"""
        min_score = min(scores.values())
        min_component = [k for k, v in scores.items() if v == min_score][0]
        
        focus_map = {
            'cpu_score': 'CPU and processing optimization',
            'memory_score': 'Memory allocation and caching',
            'storage_score': 'Storage performance and I/O optimization',
            'network_score': 'Network bandwidth and latency',
            'database_score': 'Database configuration and tuning'
        }
        
        return focus_map.get(min_component, 'General performance optimization')
    
    def _ai_identify_bottlenecks(self, cpu_perf: Dict, mem_perf: Dict, 
                               storage_perf: Dict, net_perf: Dict, config: Dict) -> Dict:
        """AI-powered bottleneck identification"""
        
        bottlenecks = []
        ai_insights = []
        
        # CPU analysis
        if cpu_perf.get('ai_bottleneck_risk') == 'high':
            bottlenecks.append("CPU performance insufficient for workload")
            ai_insights.append("AI predicts CPU will be primary bottleneck during peak loads")
        
        # Memory analysis
        if mem_perf.get('memory_pressure') == 'high':
            bottlenecks.append("Memory pressure detected")
            ai_insights.append(f"AI recommends {mem_perf['ai_memory_analysis']['recommended_memory_gb']:.0f}GB for optimal performance")
        
        # Storage analysis
        if storage_perf.get('ai_storage_analysis', {}).get('performance_impact') == 'high':
            bottlenecks.append("Storage IOPS insufficient")
            ai_insights.append("AI suggests storage upgrade will provide significant performance improvement")
        
        # Network analysis
        if net_perf.get('ai_network_analysis', {}).get('bottleneck_risk') == 'high':
            bottlenecks.append("Network bandwidth limited")
            ai_insights.append("AI identifies network as migration performance constraint")
        
        if not bottlenecks:
            bottlenecks.append("No significant bottlenecks detected by AI analysis")
            ai_insights.append("AI assessment indicates well-balanced system configuration")
        
        return {
            'bottlenecks': bottlenecks,
            'ai_insights': ai_insights
        }
    
    def _generate_ai_optimization_recommendations(self, overall_perf: Dict, 
                                                os_impact: Dict, config: Dict) -> List[str]:
        """Generate AI-powered optimization recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if overall_perf['ai_adjusted_score'] < 60:
            recommendations.append("AI recommends comprehensive performance review and hardware upgrade")
        elif overall_perf['ai_adjusted_score'] < 80:
            recommendations.append("AI suggests targeted optimization of lowest-performing components")
        
        # OS-specific recommendations
        os_insights = os_impact.get('ai_insights', {})
        if os_insights:
            recommendations.extend([
                f"OS Consideration: {insight}" for insight in os_insights.get('migration_considerations', [])[:2]
            ])
        
        # Workload-specific recommendations
        workload_opt = overall_perf.get('ai_workload_optimization', {})
        if workload_opt.get('recommended_focus'):
            recommendations.append(f"AI Priority: Focus on {workload_opt['recommended_focus']}")
        
        # Database-specific recommendations
        if config['database_size_gb'] > 10000:
            recommendations.append("AI recommends staged migration approach for large database")
        
        if config['performance_requirements'] == 'high':
            recommendations.append("AI suggests performance testing in AWS environment before full migration")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    # Keep original helper methods
    def _get_performance_tier(self, score: float) -> str:
        if score >= 80:
            return "High Performance"
        elif score >= 60:
            return "Standard Performance"
        elif score >= 40:
            return "Basic Performance"
        else:
            return "Limited Performance"
    
    def _get_ai_scaling_recommendation(self, cpu_score: float, memory_score: float, storage_score: float) -> List[str]:
        """AI-enhanced scaling recommendations"""
        recommendations = []
        
        if cpu_score < 60:
            recommendations.append("AI Priority: CPU upgrade or more cores for improved performance")
        if memory_score < 60:
            recommendations.append("AI Priority: Memory expansion for better caching efficiency")
        if storage_score < 60:
            recommendations.append("AI Priority: Storage upgrade to NVMe SSD for IOPS improvement")
        
        if not recommendations:
            recommendations.append("AI Assessment: System is well-balanced for current workload")
        
        return recommendations

class EnhancedMigrationAnalyzer:
    """Enhanced migration analyzer with AI and AWS API integration plus FSx support"""
    
    def __init__(self):
        self.ai_manager = AnthropicAIManager()
        self.aws_api = AWSAPIManager()
        self.os_manager = OSPerformanceManager()
        self.aws_manager = EnhancedAWSMigrationManager(self.aws_api, self.ai_manager)
        self.network_manager = EnhancedNetworkIntelligenceManager()
        self.agent_manager = EnhancedAgentSizingManager()
        self.onprem_analyzer = OnPremPerformanceAnalyzer()
        
        # Keep original hardware configurations
        self.nic_types = {
            'gigabit_copper': {'max_speed': 1000, 'efficiency': 0.85},
            'gigabit_fiber': {'max_speed': 1000, 'efficiency': 0.90},
            '10g_copper': {'max_speed': 10000, 'efficiency': 0.88},
            '10g_fiber': {'max_speed': 10000, 'efficiency': 0.92},
            '25g_fiber': {'max_speed': 25000, 'efficiency': 0.94},
            '40g_fiber': {'max_speed': 40000, 'efficiency': 0.95}
        }
    
    async def comprehensive_ai_migration_analysis(self, config: Dict) -> Dict:
        """Comprehensive AI-powered migration analysis with agent scaling and FSx support"""
        
        # API status tracking
        api_status = APIStatus(
            anthropic_connected=self.ai_manager.connected,
            aws_pricing_connected=self.aws_api.connected,
            last_update=datetime.now()
        )
        
        # Enhanced on-premises performance analysis
        onprem_performance = self.onprem_analyzer.calculate_ai_enhanced_performance(config, self.os_manager)
        
        # Determine network path key based on config and destination storage
        network_path_key = self._get_network_path_key(config)
        
        # AI-enhanced network path analysis
        network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
        
        # Determine migration type and tools (preserved)
        is_homogeneous = config['source_database_engine'] == config['database_engine']
        migration_type = 'homogeneous' if is_homogeneous else 'heterogeneous'
        primary_tool = 'datasync' if is_homogeneous else 'dms'
        
        # Enhanced agent analysis with scaling support and destination storage
        agent_analysis = await self._analyze_ai_migration_agents_with_scaling(config, primary_tool, network_perf)
        
        # Calculate effective migration throughput with multiple agents
        agent_throughput = agent_analysis['total_effective_throughput']
        network_throughput = network_perf['effective_bandwidth_mbps']
        migration_throughput = min(agent_throughput, network_throughput)
        
        # AI-enhanced migration time calculation with agent scaling
        migration_time_hours = await self._calculate_ai_migration_time_with_agents(
            config, migration_throughput, onprem_performance, agent_analysis
        )
        
        # AI-powered AWS sizing recommendations
        aws_sizing = await self.aws_manager.ai_enhanced_aws_sizing(config)
        
        # Enhanced cost analysis with agent scaling costs and FSx costs
        cost_analysis = await self._calculate_ai_enhanced_costs_with_agents(
            config, aws_sizing, agent_analysis, network_perf
        )
        
        # Generate FSx destination comparisons
        fsx_comparisons = await self._generate_fsx_destination_comparisons(config)
        
        return {
            'api_status': api_status,
            'onprem_performance': onprem_performance,
            'network_performance': network_perf,
            'migration_type': migration_type,
            'primary_tool': primary_tool,
            'agent_analysis': agent_analysis,
            'migration_throughput_mbps': migration_throughput,
            'estimated_migration_time_hours': migration_time_hours,
            'aws_sizing_recommendations': aws_sizing,
            'cost_analysis': cost_analysis,
            'fsx_comparisons': fsx_comparisons,
            'ai_overall_assessment': await self._generate_ai_overall_assessment_with_agents(
                config, onprem_performance, aws_sizing, migration_time_hours, agent_analysis
            )
        }
    
    def _get_network_path_key(self, config: Dict) -> str:
        """Get the appropriate network path key based on configuration"""
        
        os_type = 'linux' if 'linux' in config['operating_system'] or 'ubuntu' in config['operating_system'] or 'rhel' in config['operating_system'] else 'windows'
        environment = config['environment'].replace('-', '_')
        destination_storage = config.get('destination_storage_type', 'S3').lower()
        
        # Construct path key
        if environment == 'non_production':
            if destination_storage == 's3':
                return f"nonprod_sj_{os_type}_{'nas' if os_type == 'linux' else 'share'}_s3"
            elif destination_storage == 'fsx_windows':
                return f"nonprod_sj_{os_type}_{'nas' if os_type == 'linux' else 'share'}_fsx_windows"
            elif destination_storage == 'fsx_lustre':
                return f"nonprod_sj_{os_type}_{'nas' if os_type == 'linux' else 'share'}_fsx_lustre"
        else:  # production
            if destination_storage == 's3':
                return f"prod_sa_{os_type}_{'nas' if os_type == 'linux' else 'share'}_s3"
            elif destination_storage == 'fsx_windows':
                return f"prod_sa_{os_type}_{'nas' if os_type == 'linux' else 'share'}_fsx_windows"
            elif destination_storage == 'fsx_lustre':
                return f"prod_sa_{os_type}_{'nas' if os_type == 'linux' else 'share'}_fsx_lustre"
        
        # Default fallback
        return "nonprod_sj_linux_nas_s3"
    
    async def _generate_fsx_destination_comparisons(self, config: Dict) -> Dict:
        """Generate comprehensive FSx destination comparisons"""
        
        comparisons = {}
        destination_types = ['S3', 'FSx_Windows', 'FSx_Lustre']
        
        for dest_type in destination_types:
            # Create temporary config for this destination
            temp_config = config.copy()
            temp_config['destination_storage_type'] = dest_type
            
            # Get network path for this destination
            network_path_key = self._get_network_path_key(temp_config)
            
            try:
                # Calculate network performance
                network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
                
                # Calculate agent configuration for this destination
                is_homogeneous = config['source_database_engine'] == config['database_engine']
                primary_tool = 'datasync' if is_homogeneous else 'dms'
                agent_size = config.get('datasync_agent_size' if is_homogeneous else 'dms_agent_size', 'medium')
                num_agents = config.get('number_of_agents', 1)
                
                agent_config = self.agent_manager.calculate_agent_configuration(
                    primary_tool, agent_size, num_agents, dest_type
                )
                
                # Calculate migration time
                agent_throughput = agent_config['total_max_throughput_mbps']
                network_throughput = network_perf['effective_bandwidth_mbps']
                migration_throughput = min(agent_throughput, network_throughput)
                
                database_size_gb = config['database_size_gb']
                migration_time_hours = (database_size_gb * 8 * 1000) / (migration_throughput * 3600)
                
                # Apply complexity factors
                if config['source_database_engine'] != config['database_engine']:
                    migration_time_hours *= 1.3
                if 'windows' in config['operating_system']:
                    migration_time_hours *= 1.1
                if config['server_type'] == 'vmware':
                    migration_time_hours *= 1.05
                
                # Calculate costs (basic estimation)
                storage_multiplier = {
                    'S3': 1.0,
                    'FSx_Windows': 2.5,
                    'FSx_Lustre': 4.0
                }.get(dest_type, 1.0)
                
                base_storage_cost = database_size_gb * 0.023  # S3 standard pricing
                estimated_storage_cost = base_storage_cost * storage_multiplier
                
                comparisons[dest_type] = {
                    'destination_type': dest_type,
                    'network_performance': network_perf,
                    'agent_configuration': agent_config,
                    'migration_throughput_mbps': migration_throughput,
                    'estimated_migration_time_hours': migration_time_hours,
                    'estimated_monthly_storage_cost': estimated_storage_cost,
                    'performance_rating': self._calculate_destination_performance_rating(dest_type, network_perf, agent_config),
                    'cost_rating': self._calculate_destination_cost_rating(dest_type, estimated_storage_cost),
                    'complexity_rating': self._calculate_destination_complexity_rating(dest_type, config),
                    'recommendations': self._get_destination_recommendations(dest_type, config, network_perf)
                }
                
            except Exception as e:
                logger.warning(f"Failed to analyze destination {dest_type}: {e}")
                # Provide fallback data
                comparisons[dest_type] = {
                    'destination_type': dest_type,
                    'error': str(e),
                    'estimated_migration_time_hours': 24,
                    'performance_rating': 'Unknown',
                    'cost_rating': 'Unknown',
                    'complexity_rating': 'Unknown'
                }
        
        return comparisons
    
    def _calculate_destination_performance_rating(self, dest_type: str, network_perf: Dict, agent_config: Dict) -> str:
        """Calculate performance rating for destination type"""
        
        throughput = agent_config.get('total_max_throughput_mbps', 0)
        network_quality = network_perf.get('ai_enhanced_quality_score', 0)
        
        # Base performance scoring
        performance_score = 0
        
        if dest_type == 'S3':
            performance_score = 70 + (network_quality * 0.3)
        elif dest_type == 'FSx_Windows':
            performance_score = 80 + (network_quality * 0.2)
        elif dest_type == 'FSx_Lustre':
            performance_score = 95 + (network_quality * 0.05)
        
        # Adjust for throughput
        if throughput > 2000:
            performance_score += 10
        elif throughput > 1000:
            performance_score += 5
        
        if performance_score >= 90:
            return "Excellent"
        elif performance_score >= 80:
            return "Very Good"
        elif performance_score >= 70:
            return "Good"
        elif performance_score >= 60:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_destination_cost_rating(self, dest_type: str, estimated_cost: float) -> str:
        """Calculate cost rating for destination type"""
        
        if dest_type == 'S3':
            return "Excellent"
        elif dest_type == 'FSx_Windows':
            return "Good"
        elif dest_type == 'FSx_Lustre':
            return "Fair"
        else:
            return "Unknown"
    
    def _calculate_destination_complexity_rating(self, dest_type: str, config: Dict) -> str:
        """Calculate complexity rating for destination type"""
        
        base_complexity = 1
        
        if config['source_database_engine'] != config['database_engine']:
            base_complexity += 1
        if config['database_size_gb'] > 10000:
            base_complexity += 1
        if config.get('number_of_agents', 1) > 3:
            base_complexity += 1
        
        # Destination-specific complexity
        if dest_type == 'S3':
            dest_complexity = 1
        elif dest_type == 'FSx_Windows':
            dest_complexity = 2
        elif dest_type == 'FSx_Lustre':
            dest_complexity = 3
        else:
            dest_complexity = 1
        
        total_complexity = base_complexity + dest_complexity
        
        if total_complexity <= 2:
            return "Low"
        elif total_complexity <= 4:
            return "Medium"
        else:
            return "High"
    
    def _get_destination_recommendations(self, dest_type: str, config: Dict, network_perf: Dict) -> List[str]:
        """Get recommendations for specific destination type"""
        
        recommendations = []
        
        if dest_type == 'S3':
            recommendations.extend([
                "Ideal for cost-effective cloud storage",
                "Simple integration with AWS services",
                "Excellent durability and availability",
                "Consider S3 Intelligent Tiering for cost optimization"
            ])
        elif dest_type == 'FSx_Windows':
            recommendations.extend([
                "Perfect for Windows-based applications",
                "Native Windows file system features",
                "Better performance than S3 for file-based workloads",
                "Requires Active Directory integration planning"
            ])
        elif dest_type == 'FSx_Lustre':
            recommendations.extend([
                "Best choice for high-performance computing",
                "Extremely high throughput and low latency",
                "Ideal for analytics and machine learning workloads",
                "Requires Lustre expertise and careful configuration"
            ])
        
        # Add configuration-specific recommendations
        if config['database_size_gb'] > 20000:
            if dest_type == 'FSx_Lustre':
                recommendations.append("Large database size benefits from Lustre's parallel performance")
            elif dest_type == 'S3':
                recommendations.append("Consider multipart uploads for large database migration")
        
        if config['performance_requirements'] == 'high':
            if dest_type == 'FSx_Lustre':
                recommendations.append("High performance requirements perfectly match Lustre capabilities")
            elif dest_type == 'S3':
                recommendations.append("Consider S3 Transfer Acceleration for better performance")
        
        return recommendations[:4]  # Limit to top 4 recommendations
    
    async def _analyze_ai_migration_agents_with_scaling(self, config: Dict, primary_tool: str, network_perf: Dict) -> Dict:
        """Enhanced migration agent analysis with scaling support and destination storage"""
        
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        
        if primary_tool == 'datasync':
            agent_size = config['datasync_agent_size']
            agent_config = self.agent_manager.calculate_agent_configuration('datasync', agent_size, num_agents, destination_storage)
        else:
            agent_size = config['dms_agent_size']
            agent_config = self.agent_manager.calculate_agent_configuration('dms', agent_size, num_agents, destination_storage)
        
        # Calculate throughput impact with scaling and destination storage
        total_max_throughput = agent_config['total_max_throughput_mbps']
        network_bandwidth = network_perf['effective_bandwidth_mbps']
        total_effective_throughput = min(total_max_throughput, network_bandwidth)
        throughput_impact = total_effective_throughput / total_max_throughput
        
        # Determine bottleneck with agent scaling considerations
        if total_max_throughput < network_bandwidth:
            bottleneck = f'agents ({num_agents} agents)'
            bottleneck_severity = 'high' if throughput_impact < 0.7 else 'medium'
        else:
            bottleneck = 'network'
            bottleneck_severity = 'medium' if throughput_impact > 0.8 else 'high'
        
        # AI enhancement: optimization recommendations with scaling and destination storage
        ai_agent_optimization = await self._get_ai_agent_optimization_with_scaling(
            agent_config, network_perf, config, num_agents
        )
        
        # Get optimal agent recommendations for this destination
        optimal_recommendations = self.agent_manager.recommend_optimal_agents(
            config['database_size_gb'],
            network_bandwidth,
            config.get('target_migration_hours', 24),
            destination_storage
        )
        
        return {
            'primary_tool': primary_tool,
            'agent_size': agent_size,
            'number_of_agents': num_agents,
            'destination_storage': destination_storage,
            'agent_configuration': agent_config,
            'total_max_throughput_mbps': total_max_throughput,
            'total_effective_throughput': total_effective_throughput,
            'throughput_impact': throughput_impact,
            'bottleneck': bottleneck,
            'bottleneck_severity': bottleneck_severity,
            'scaling_efficiency': agent_config['scaling_efficiency'],
            'management_overhead': agent_config['management_overhead_factor'],
            'storage_performance_multiplier': agent_config.get('storage_performance_multiplier', 1.0),
            'ai_optimization': ai_agent_optimization,
            'optimal_recommendations': optimal_recommendations,
            'cost_per_hour': agent_config['effective_cost_per_hour'],
            'monthly_cost': agent_config['total_monthly_cost']
        }
    
    async def _calculate_ai_migration_time_with_agents(self, config: Dict, migration_throughput: float, 
                                                     onprem_performance: Dict, agent_analysis: Dict) -> float:
        """AI-enhanced migration time calculation with agent scaling and destination storage"""
        
        database_size_gb = config['database_size_gb']
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        
        # Base calculation (preserved)
        base_time_hours = (database_size_gb * 8 * 1000) / (migration_throughput * 3600)
        
        # AI adjustments
        ai_complexity = onprem_performance.get('ai_optimization_recommendations', [])
        complexity_factor = 1.0
        
        # Increase time for complex scenarios
        if config['source_database_engine'] != config['database_engine']:
            complexity_factor *= 1.3  # Heterogeneous migration
        
        if 'windows' in config['operating_system']:
            complexity_factor *= 1.1  # Windows overhead
        
        if config['server_type'] == 'vmware':
            complexity_factor *= 1.05  # Virtualization overhead
        
        # Destination storage adjustments
        if destination_storage == 'FSx_Windows':
            complexity_factor *= 0.9  # Better performance, less time
        elif destination_storage == 'FSx_Lustre':
            complexity_factor *= 0.7  # Much better performance, significantly less time
        
        # Agent scaling adjustments
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0)
        storage_multiplier = agent_analysis.get('storage_performance_multiplier', 1.0)
        
        if num_agents > 1:
            # More agents can reduce time but with coordination overhead
            agent_time_factor = (1 / min(num_agents * scaling_efficiency * storage_multiplier, 6.0))  # Diminishing returns
            complexity_factor *= agent_time_factor
            
            # Add coordination overhead for many agents
            if num_agents > 5:
                complexity_factor *= 1.1  # 10% coordination overhead
        
        # AI insights factor
        if len(ai_complexity) > 3:
            complexity_factor *= 1.2  # Many optimization needs
        
        return base_time_hours * complexity_factor
    
    async def _calculate_ai_enhanced_costs_with_agents(self, config: Dict, aws_sizing: Dict, 
                                                     agent_analysis: Dict, network_perf: Dict) -> Dict:
        """AI-enhanced cost calculation with agent scaling costs and FSx storage costs"""
        
        # Get deployment recommendation
        deployment_rec = aws_sizing['deployment_recommendation']['recommendation']
        
        # Use real-time pricing data
        if deployment_rec == 'rds':
            aws_compute_cost = aws_sizing['rds_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['rds_recommendations']['monthly_storage_cost']
        else:
            aws_compute_cost = aws_sizing['ec2_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['ec2_recommendations']['monthly_storage_cost']
        
        # Enhanced agent costs with scaling
        agent_monthly_cost = agent_analysis.get('monthly_cost', 0)
        management_overhead = agent_analysis.get('management_overhead', 1.0)
        storage_management_overhead = agent_analysis.get('storage_management_overhead', 1.0)
        total_agent_cost = agent_monthly_cost * management_overhead * storage_management_overhead
        
        # Destination storage costs
        destination_storage = config.get('destination_storage_type', 'S3')
        destination_storage_cost = self._calculate_destination_storage_cost(config, destination_storage)
        
        # Network costs with AI optimization
        base_network_cost = 800 if 'prod' in config.get('network_path', '') else 400
        ai_optimization_factor = network_perf.get('ai_optimization_potential', 0) / 100
        optimized_network_cost = base_network_cost * (1 - ai_optimization_factor * 0.2)
        
        # OS licensing with AI insights
        os_licensing_cost = self.os_manager.operating_systems[config['operating_system']]['licensing_cost_factor'] * 150
        
        # Management costs (AI reduces if RDS, increases with more agents and complex storage)
        base_management_cost = 200 if deployment_rec == 'ec2' else 50
        ai_management_reduction = 0.15 if aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6) < 5 else 0
        agent_management_increase = (config.get('number_of_agents', 1) - 1) * 50  # $50 per additional agent management
        storage_management_increase = {
            'S3': 0,
            'FSx_Windows': 100,
            'FSx_Lustre': 200
        }.get(destination_storage, 0)
        
        management_cost = ((base_management_cost * (1 - ai_management_reduction)) + 
                          agent_management_increase + storage_management_increase)
        
        # One-time migration costs with AI complexity, agent scaling, and storage complexity
        ai_complexity = aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.1
        
        # Agent scaling impact on migration cost
        num_agents = config.get('number_of_agents', 1)
        agent_setup_cost = num_agents * 500  # $500 setup cost per agent
        agent_coordination_cost = max(0, (num_agents - 1) * 200)  # $200 coordination cost per additional agent
        
        # Destination storage setup costs
        storage_setup_costs = {
            'S3': 100,
            'FSx_Windows': 1000,
            'FSx_Lustre': 2000
        }
        storage_setup_cost = storage_setup_costs.get(destination_storage, 100)
        
        base_migration_cost = config['database_size_gb'] * 0.1
        one_time_migration_cost = (base_migration_cost * complexity_multiplier + 
                                 agent_setup_cost + agent_coordination_cost + storage_setup_cost)
        
        total_monthly_cost = (aws_compute_cost + aws_storage_cost + total_agent_cost + 
                            destination_storage_cost + optimized_network_cost + 
                            os_licensing_cost + management_cost)
        
        # AI-predicted savings with agent optimization and storage efficiency
        ai_optimization_potential = aws_sizing.get('ai_analysis', {}).get('performance_recommendations', [])
        agent_efficiency_bonus = agent_analysis.get('scaling_efficiency', 1.0) * 0.1  # Up to 10% bonus for efficient scaling
        storage_efficiency_bonus = {
            'S3': 0.15,  # High efficiency
            'FSx_Windows': 0.10,  # Medium efficiency
            'FSx_Lustre': 0.05   # Lower efficiency due to high performance focus
        }.get(destination_storage, 0.10)
        
        estimated_monthly_savings = total_monthly_cost * (0.1 + len(ai_optimization_potential) * 0.02 + 
                                                        agent_efficiency_bonus + storage_efficiency_bonus)
        roi_months = int(one_time_migration_cost / estimated_monthly_savings) if estimated_monthly_savings > 0 else None
        
        return {
            'aws_compute_cost': aws_compute_cost,
            'aws_storage_cost': aws_storage_cost,
            'agent_cost': total_agent_cost,
            'agent_base_cost': agent_monthly_cost,
            'agent_management_overhead': management_overhead,
            'destination_storage_cost': destination_storage_cost,
            'destination_storage_type': destination_storage,
            'network_cost': optimized_network_cost,
            'os_licensing_cost': os_licensing_cost,
            'management_cost': management_cost,
            'total_monthly_cost': total_monthly_cost,
            'one_time_migration_cost': one_time_migration_cost,
            'agent_setup_cost': agent_setup_cost,
            'agent_coordination_cost': agent_coordination_cost,
            'storage_setup_cost': storage_setup_cost,
            'estimated_monthly_savings': estimated_monthly_savings,
            'roi_months': roi_months,
            'ai_cost_insights': {
                'ai_optimization_factor': ai_optimization_factor,
                'complexity_multiplier': complexity_multiplier,
                'management_reduction': ai_management_reduction,
                'agent_efficiency_bonus': agent_efficiency_bonus,
                'storage_efficiency_bonus': storage_efficiency_bonus,
                'potential_additional_savings': f"{len(ai_optimization_potential) * 2 + int(agent_efficiency_bonus * 10) + int(storage_efficiency_bonus * 10)}% through AI, agent, and storage optimization"
            }
        }
    
    def _calculate_destination_storage_cost(self, config: Dict, destination_storage: str) -> float:
        """Calculate monthly cost for destination storage"""
        
        database_size_gb = config['database_size_gb']
        
        # Base cost per GB per month
        storage_costs = {
            'S3': 0.023,  # S3 Standard
            'FSx_Windows': 0.13,  # FSx for Windows
            'FSx_Lustre': 0.14   # FSx for Lustre
        }
        
        base_cost_per_gb = storage_costs.get(destination_storage, 0.023)
        
        # Storage multiplier based on migration requirements
        storage_multiplier = 1.5  # Extra space for migration
        
        return database_size_gb * storage_multiplier * base_cost_per_gb
    
    async def _get_ai_agent_optimization_with_scaling(self, agent_config: Dict, network_perf: Dict, 
                                                    config: Dict, num_agents: int) -> Dict:
        """Get AI-powered agent optimization recommendations with scaling and destination storage considerations"""
        
        # Enhanced analysis with scaling and destination storage
        network_bandwidth = network_perf['effective_bandwidth_mbps']
        total_agent_capacity = agent_config['total_max_throughput_mbps']
        destination_storage = config.get('destination_storage_type', 'S3')
        
        if network_bandwidth < total_agent_capacity:
            bottleneck_type = 'network'
            optimization_potential = network_perf.get('ai_optimization_potential', 0)
            recommendations = network_perf.get('ai_insights', {}).get('optimization_opportunities', [])
        else:
            bottleneck_type = 'agents'
            optimization_potential = max(0, 20 - (agent_config['scaling_efficiency'] * 20))  # Efficiency-based optimization
            recommendations = agent_config.get('scaling_recommendations', [])
        
        # Agent-specific optimizations
        agent_recommendations = [
            f"Optimize {num_agents}-agent coordination and load balancing",
            "Implement intelligent retry mechanisms across agents",
            "Configure bandwidth throttling during peak hours"
        ]
        
        if num_agents > 3:
            agent_recommendations.append("Consider agent consolidation to reduce complexity")
        elif num_agents == 1 and config['database_size_gb'] > 5000:
            agent_recommendations.append("Scale to multiple agents for better throughput")
        
        # Destination storage-specific optimizations
        if destination_storage == 'FSx_Windows':
            agent_recommendations.append("Configure agents for SMB protocol optimization")
            agent_recommendations.append("Implement Windows file sharing best practices")
        elif destination_storage == 'FSx_Lustre':
            agent_recommendations.append("Optimize agents for Lustre parallel I/O")
            agent_recommendations.append("Configure Lustre striping for maximum performance")
        
        recommendations.extend(agent_recommendations)
        
        return {
            'bottleneck_type': bottleneck_type,
            'optimization_potential_percent': optimization_potential,
            'recommendations': recommendations[:6],
            'estimated_improvement': f"{optimization_potential}% throughput improvement possible",
            'scaling_assessment': agent_config.get('optimal_configuration', {}),
            'current_efficiency': agent_config.get('scaling_efficiency', 1.0) * 100,
            'destination_storage_impact': f"{destination_storage} provides {agent_config.get('storage_performance_multiplier', 1.0):.1f}x performance multiplier"
        }
    
    async def _generate_ai_overall_assessment_with_agents(self, config: Dict, onprem_performance: Dict, 
                                                        aws_sizing: Dict, migration_time: float, 
                                                        agent_analysis: Dict) -> Dict:
        """Generate AI-powered overall migration assessment with agent scaling and destination storage considerations"""
        
        # Migration readiness score with agent and destination storage considerations
        readiness_factors = []
        readiness_score = 100
        
        # Performance readiness
        perf_score = onprem_performance.get('performance_score', 0)
        if perf_score < 50:
            readiness_score -= 20
            readiness_factors.append("Performance optimization needed before migration")
        elif perf_score < 70:
            readiness_score -= 10
            readiness_factors.append("Minor performance improvements recommended")
        
        # Complexity assessment
        ai_complexity = aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6)
        if ai_complexity > 8:
            readiness_score -= 25
            readiness_factors.append("High complexity migration requires extensive planning")
        elif ai_complexity > 6:
            readiness_score -= 10
            readiness_factors.append("Moderate complexity migration needs careful execution")
        
        # Agent scaling assessment
        num_agents = config.get('number_of_agents', 1)
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0)
        optimal_config = agent_analysis.get('optimal_recommendations', {}).get('optimal_configuration')
        
        if optimal_config and num_agents != optimal_config['configuration']['number_of_agents']:
            readiness_score -= 10
            readiness_factors.append(f"Agent configuration not optimal - consider {optimal_config['configuration']['number_of_agents']} agents")
        
        if scaling_efficiency < 0.85:
            readiness_score -= 15
            readiness_factors.append("Agent scaling efficiency below optimal - coordination overhead detected")
        
        # Destination storage assessment
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Lustre':
            readiness_score += 5  # Bonus for high-performance storage
            readiness_factors.append("FSx for Lustre provides excellent performance capabilities")
        elif destination_storage == 'FSx_Windows':
            readiness_score += 3  # Small bonus for enhanced performance
            readiness_factors.append("FSx for Windows provides good performance and integration")
        
        # Time assessment
        if migration_time > 48:
            readiness_score -= 15
            readiness_factors.append("Long migration time requires extensive downtime planning")
        elif migration_time > 24:
            readiness_score -= 5
            readiness_factors.append("Extended migration window needed")
        
        # Network readiness
        network_quality = onprem_performance.get('network_performance', {}).get('ai_optimization_score', 80)
        if network_quality < 60:
            readiness_score -= 15
            readiness_factors.append("Network optimization required for successful migration")
        
        # Agent bottleneck assessment
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        if bottleneck_severity == 'high':
            readiness_score -= 20
            readiness_factors.append("Significant agent or network bottleneck detected")
        elif bottleneck_severity == 'medium':
            readiness_score -= 5
            readiness_factors.append("Minor bottleneck may impact migration performance")
        
        # Success probability with agent scaling bonus and destination storage bonus
        base_success_probability = max(60, min(95, readiness_score))
        agent_efficiency_bonus = (scaling_efficiency - 0.8) * 25 if scaling_efficiency > 0.8 else 0  # Up to 5% bonus
        storage_performance_bonus = {
            'S3': 0,
            'FSx_Windows': 2,
            'FSx_Lustre': 5
        }.get(destination_storage, 0)
        
        success_probability = min(95, base_success_probability + agent_efficiency_bonus + storage_performance_bonus)
        
        # Risk level
        if readiness_score >= 80:
            risk_level = "Low"
        elif readiness_score >= 65:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'migration_readiness_score': readiness_score,
            'success_probability': success_probability,
            'risk_level': risk_level,
            'readiness_factors': readiness_factors,
            'ai_confidence': aws_sizing.get('deployment_recommendation', {}).get('confidence', 0.5),
            'agent_scaling_impact': {
                'scaling_efficiency': scaling_efficiency * 100,
                'optimal_agents': optimal_config['configuration']['number_of_agents'] if optimal_config else num_agents,
                'current_agents': num_agents,
                'efficiency_bonus': agent_efficiency_bonus
            },
            'destination_storage_impact': {
                'storage_type': destination_storage,
                'performance_bonus': storage_performance_bonus,
                'storage_performance_multiplier': agent_analysis.get('storage_performance_multiplier', 1.0)
            },
            'recommended_next_steps': self._get_next_steps_with_agents(readiness_score, ai_complexity, agent_analysis, destination_storage),
            'timeline_recommendation': self._get_timeline_recommendation_with_agents(migration_time, ai_complexity, num_agents, destination_storage)
        }
    
    def _get_next_steps_with_agents(self, readiness_score: float, ai_complexity: int, 
                                  agent_analysis: Dict, destination_storage: str) -> List[str]:
        """Get recommended next steps with agent scaling and destination storage considerations"""
        
        steps = []
        
        if readiness_score < 70:
            steps.append("Conduct detailed performance baseline and optimization")
            steps.append("Address identified bottlenecks before migration")
        
        if ai_complexity > 7:
            steps.append("Develop comprehensive migration strategy and testing plan")
            steps.append("Consider engaging AWS migration specialists")
        
        # Agent-specific steps
        num_agents = agent_analysis.get('number_of_agents', 1)
        optimal_config = agent_analysis.get('optimal_recommendations', {}).get('optimal_configuration')
        
        if optimal_config and num_agents != optimal_config['configuration']['number_of_agents']:
            steps.append(f"Optimize agent configuration to {optimal_config['configuration']['number_of_agents']} agents")
        
        if num_agents > 3:
            steps.append(f"Set up centralized monitoring for {num_agents}-agent coordination")
        
        # Destination storage-specific steps
        if destination_storage == 'FSx_Windows':
            steps.append("Plan Active Directory integration for FSx for Windows")
            steps.append("Test SMB protocol performance and optimization")
        elif destination_storage == 'FSx_Lustre':
            steps.append("Design Lustre file system layout and striping strategy")
            steps.append("Plan Lustre client configuration and optimization")
        
        steps.extend([
            "Set up AWS environment and conduct connectivity tests",
            f"Perform proof-of-concept migration with {destination_storage} destination",
            "Develop detailed cutover and rollback procedures"
        ])
        
        return steps[:6]
    
    def _get_timeline_recommendation_with_agents(self, migration_time: float, ai_complexity: int, 
                                               num_agents: int, destination_storage: str) -> Dict:
        """Get AI-recommended timeline with agent scaling and destination storage considerations"""
        
        # Base phases
        planning_weeks = 2 + (ai_complexity - 5) * 0.5
        testing_weeks = 3 + (ai_complexity - 5) * 0.5
        migration_hours = migration_time
        
        # Agent scaling adjustments
        if num_agents > 3:
            planning_weeks += 1  # More planning for complex agent setups
            testing_weeks += 0.5  # Additional testing for coordination
        
        # Destination storage adjustments
        if destination_storage == 'FSx_Windows':
            planning_weeks += 0.5  # AD integration planning
            testing_weeks += 0.5   # SMB testing
        elif destination_storage == 'FSx_Lustre':
            planning_weeks += 1.0  # Lustre design and planning
            testing_weeks += 1.0   # Lustre performance testing
        
        return {
            'planning_phase_weeks': max(1, planning_weeks),
            'testing_phase_weeks': max(2, testing_weeks),
            'migration_window_hours': migration_hours,
            'total_project_weeks': max(6, planning_weeks + testing_weeks + 1),
            'recommended_approach': 'staged' if ai_complexity > 7 or migration_time > 24 or num_agents > 5 else 'direct',
            'agent_coordination_time': f"{num_agents * 2} hours for agent setup and coordination" if num_agents > 1 else "N/A",
            'storage_setup_time': {
                'S3': "Minimal setup time",
                'FSx_Windows': f"{planning_weeks * 2} hours for AD integration and FSx setup",
                'FSx_Lustre': f"{planning_weeks * 4} hours for Lustre configuration and optimization"
            }.get(destination_storage, "Standard setup time")
        }

# PDF Report Generation Class
class PDFReportGenerator:
    """Generate executive PDF reports for migration analysis"""
    
    def __init__(self):
        self.report_style = {
            'title_color': '#1e3c72',
            'header_color': '#2a5298', 
            'accent_color': '#3498db',
            'text_color': '#2c3e50',
            'background_color': '#f8f9fa'
        }
    
    def generate_executive_report(self, analysis: Dict, config: Dict) -> bytes:
        """Generate comprehensive executive PDF report"""
        
        # Create matplotlib figure for PDF
        plt.style.use('default')
        
        # Create PDF buffer
        buffer = BytesIO()
        
        with PdfPages(buffer) as pdf:
            # Page 1: Executive Summary
            self._create_executive_summary_page(pdf, analysis, config)
            
            # Page 2: Technical Analysis
            self._create_technical_analysis_page(pdf, analysis, config)
            
            # Page 3: AWS Sizing Recommendations
            self._create_aws_sizing_page(pdf, analysis, config)
            
            # Page 4: Cost Analysis
            self._create_cost_analysis_page(pdf, analysis, config)
            
            # Page 5: Risk Assessment & Timeline
            self._create_risk_timeline_page(pdf, analysis, config)
            
            # Page 6: Agent Scaling Analysis
            self._create_agent_scaling_page(pdf, analysis, config)
            
            # Page 7: FSx Destination Comparison (New)
            self._create_fsx_comparison_page(pdf, analysis, config)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_executive_summary_page(self, pdf, analysis: Dict, config: Dict):
        """Create executive summary page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('AWS Database Migration - Executive Summary', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        # Migration Overview
        ax1.axis('off')
        ax1.text(0.05, 0.95, 'Migration Overview', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax1.transAxes)
        
        overview_text = f"""
        Source: {config['source_database_engine'].upper()} ({config['database_size_gb']:,} GB)
        Target: AWS {config['database_engine'].upper()}
        Type: {'Homogeneous' if config['source_database_engine'] == config['database_engine'] else 'Heterogeneous'}
        Environment: {config['environment'].title()}
        Destination: {config.get('destination_storage_type', 'S3')}
        Migration Time: {analysis.get('estimated_migration_time_hours', 0):.1f} hours
        Downtime Tolerance: {config['downtime_tolerance_minutes']} minutes
        Agents: {config.get('number_of_agents', 1)} {analysis.get('primary_tool', 'DataSync').upper()} agents
        """
        
        ax1.text(0.05, 0.75, overview_text, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        
        # AI Readiness Score (Gauge chart)
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        self._create_gauge_chart(ax2, readiness_score, 'Migration Readiness', 'AI Assessment')
        
        # Cost Summary
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Cost Summary', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        cost_analysis = analysis.get('cost_analysis', {})
        cost_text = f"""
        Monthly AWS Cost: ${cost_analysis.get('total_monthly_cost', 0):,.0f}
        One-time Migration: ${cost_analysis.get('one_time_migration_cost', 0):,.0f}
        Annual Cost: ${cost_analysis.get('total_monthly_cost', 0) * 12:,.0f}
        Agent Costs: ${cost_analysis.get('agent_cost', 0):,.0f}/month
        Storage Destination: ${cost_analysis.get('destination_storage_cost', 0):,.0f}/month
        Potential Savings: ${cost_analysis.get('estimated_monthly_savings', 0):,.0f}/month
        ROI Timeline: {cost_analysis.get('roi_months', 'TBD')} months
        """
        
        ax3.text(0.05, 0.75, cost_text, fontsize=10, transform=ax3.transAxes, verticalalignment='top')
        
        # Performance Score
        perf_score = analysis.get('onprem_performance', {}).get('performance_score', 0)
        self._create_gauge_chart(ax4, perf_score, 'Performance Score', 'Current System')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_fsx_comparison_page(self, pdf, analysis: Dict, config: Dict):
        """Create FSx destination comparison page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('FSx Destination Storage Comparison Analysis', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        fsx_comparisons = analysis.get('fsx_comparisons', {})
        
        # Performance Comparison
        destinations = list(fsx_comparisons.keys())
        migration_times = [fsx_comparisons[dest].get('estimated_migration_time_hours', 0) for dest in destinations]
        
        bars = ax1.bar(destinations, migration_times, color=['#3498db', '#e74c3c', '#f39c12'])
        ax1.set_title('Migration Time Comparison', fontweight='bold')
        ax1.set_ylabel('Hours')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(migration_times) * 0.01,
                   f'{height:.1f}h', ha='center', va='bottom', fontsize=8)
        
        # Cost Comparison
        storage_costs = [fsx_comparisons[dest].get('estimated_monthly_storage_cost', 0) for dest in destinations]
        
        bars = ax2.bar(destinations, storage_costs, color=['#27ae60', '#e67e22', '#9b59b6'])
        ax2.set_title('Monthly Storage Cost Comparison', fontweight='bold')
        ax2.set_ylabel('Cost ($)')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(storage_costs) * 0.01,
                   f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # Performance vs Cost Matrix
        performance_ratings = [fsx_comparisons[dest].get('performance_rating', 'Unknown') for dest in destinations]
        cost_ratings = [fsx_comparisons[dest].get('cost_rating', 'Unknown') for dest in destinations]
        
        # Convert ratings to numeric values for plotting
        rating_map = {'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1, 'Unknown': 0}
        perf_numeric = [rating_map.get(rating, 0) for rating in performance_ratings]
        cost_numeric = [rating_map.get(rating, 0) for rating in cost_ratings]
        
        colors = ['#3498db', '#e74c3c', '#f39c12']
        for i, dest in enumerate(destinations):
            ax3.scatter(cost_numeric[i], perf_numeric[i], s=200, c=colors[i], alpha=0.7, label=dest)
            ax3.annotate(dest, (cost_numeric[i], perf_numeric[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Cost Rating')
        ax3.set_ylabel('Performance Rating')
        ax3.set_title('Performance vs Cost Matrix', fontweight='bold')
        ax3.set_xticks(range(1, 6))
        ax3.set_xticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        ax3.set_yticks(range(1, 6))
        ax3.set_yticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        ax3.grid(True, alpha=0.3)
        
        # Recommendation Summary
        ax4.axis('off')
        ax4.text(0.05, 0.95, 'Destination Recommendations', fontsize=14, fontweight='bold', 
                color=self.report_style['header_color'], transform=ax4.transAxes)
        
        # Current destination
        current_dest = config.get('destination_storage_type', 'S3')
        current_comparison = fsx_comparisons.get(current_dest, {})
        
        rec_text = f"""
        Current Selection: {current_dest}
        Performance Rating: {current_comparison.get('performance_rating', 'Unknown')}
        Cost Rating: {current_comparison.get('cost_rating', 'Unknown')}
        Complexity: {current_comparison.get('complexity_rating', 'Unknown')}
        Migration Time: {current_comparison.get('estimated_migration_time_hours', 0):.1f} hours
        
        Alternative Options:
        """
        
        # Add alternatives
        for dest in destinations:
            if dest != current_dest:
                comp = fsx_comparisons.get(dest, {})
                rec_text += f"\n{dest}: {comp.get('performance_rating', 'Unknown')} perf, {comp.get('cost_rating', 'Unknown')} cost"
        
        ax4.text(0.05, 0.85, rec_text, fontsize=9, transform=ax4.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_agent_scaling_page(self, pdf, analysis: Dict, config: Dict):
        """Create agent scaling analysis page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Agent Scaling Analysis & Optimization', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        agent_analysis = analysis.get('agent_analysis', {})
        
        # Agent Configuration Overview
        ax1.axis('off')
        ax1.text(0.05, 0.95, 'Agent Configuration', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax1.transAxes)
        
        agent_text = f"""
        Agent Type: {agent_analysis.get('primary_tool', 'Unknown').upper()}
        Agent Size: {agent_analysis.get('agent_size', 'Unknown')}
        Number of Agents: {agent_analysis.get('number_of_agents', 1)}
        Destination: {agent_analysis.get('destination_storage', 'Unknown')}
        Total Throughput: {agent_analysis.get('total_max_throughput_mbps', 0):,.0f} Mbps
        Effective Throughput: {agent_analysis.get('total_effective_throughput', 0):,.0f} Mbps
        Scaling Efficiency: {agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%
        Storage Multiplier: {agent_analysis.get('storage_performance_multiplier', 1.0):.1f}x
        Bottleneck: {agent_analysis.get('bottleneck', 'Unknown')}
        Monthly Cost: ${agent_analysis.get('monthly_cost', 0):,.0f}
        """
        
        ax1.text(0.05, 0.75, agent_text, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        
        # Throughput Comparison
        throughput_data = {
            'Max Capacity': agent_analysis.get('total_max_throughput_mbps', 0),
            'Effective Throughput': agent_analysis.get('total_effective_throughput', 0),
            'Network Limit': analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0)
        }
        
        self._create_bar_chart(ax2, throughput_data, 'Throughput Analysis', 'Mbps')
        
        # Scaling Efficiency
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0) * 100
        self._create_gauge_chart(ax3, scaling_efficiency, 'Scaling Efficiency', f"{agent_analysis.get('number_of_agents', 1)} Agents")
        
        # Cost vs Performance
        optimal_config = agent_analysis.get('optimal_recommendations', {}).get('optimal_configuration')
        if optimal_config:
            cost_perf_data = {
                'Current Config': agent_analysis.get('monthly_cost', 0),
                'Optimal Config': optimal_config.get('total_cost_per_hour', 0) * 24 * 30
            }
            self._create_bar_chart(ax4, cost_perf_data, 'Cost Comparison', 'Monthly Cost ($)')
        else:
            ax4.text(0.5, 0.5, 'Optimal Configuration\nData Not Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Include all other PDF generation methods from original code...
    def _create_technical_analysis_page(self, pdf, analysis: Dict, config: Dict):
        """Create technical analysis page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Technical Analysis & Performance Assessment', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        # Performance Breakdown
        onprem_perf = analysis.get('onprem_performance', {}).get('overall_performance', {})
        
        performance_metrics = {
            'CPU': onprem_perf.get('cpu_score', 0),
            'Memory': onprem_perf.get('memory_score', 0),
            'Storage': onprem_perf.get('storage_score', 0),
            'Network': onprem_perf.get('network_score', 0),
            'Database': onprem_perf.get('database_score', 0)
        }
        
        # Performance radar chart
        self._create_radar_chart(ax1, performance_metrics, 'Current Performance Profile')
        
        # Network Analysis
        network_perf = analysis.get('network_performance', {})
        network_data = {
            'Quality Score': network_perf.get('network_quality_score', 0),
            'AI Enhanced': network_perf.get('ai_enhanced_quality_score', 0),
            'Bandwidth (%)': min(100, network_perf.get('effective_bandwidth_mbps', 0) / 100),
            'Reliability (%)': network_perf.get('total_reliability', 0) * 100
        }
        
        self._create_bar_chart(ax2, network_data, 'Network Performance Analysis', 'Score/Percentage')
        
        # OS Performance Impact
        os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
        os_data = {
            'CPU Efficiency': os_impact.get('cpu_efficiency', 0) * 100,
            'Memory Efficiency': os_impact.get('memory_efficiency', 0) * 100,
            'I/O Efficiency': os_impact.get('io_efficiency', 0) * 100,
            'Network Efficiency': os_impact.get('network_efficiency', 0) * 100,
            'DB Optimization': os_impact.get('db_optimization', 0) * 100
        }
        
        self._create_bar_chart(ax3, os_data, 'OS Performance Impact', 'Efficiency (%)')
        
        # Bottleneck Analysis
        ax4.axis('off')
        ax4.text(0.05, 0.95, 'Identified Bottlenecks', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax4.transAxes)
        
        bottlenecks = analysis.get('onprem_performance', {}).get('bottlenecks', [])
        ai_insights = analysis.get('onprem_performance', {}).get('ai_insights', [])
        
        bottleneck_text = "Current Bottlenecks:\n"
        for i, bottleneck in enumerate(bottlenecks[:3], 1):
            bottleneck_text += f"{i}. {bottleneck}\n"
        
        bottleneck_text += "\nAI Insights:\n"
        for i, insight in enumerate(ai_insights[:2], 1):
            bottleneck_text += f"{i}. {insight}\n"
        
        ax4.text(0.05, 0.85, bottleneck_text, fontsize=9, transform=ax4.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_aws_sizing_page(self, pdf, analysis: Dict, config: Dict):
        """Create AWS sizing recommendations page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('AWS Sizing & Configuration Recommendations', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        deployment_rec = aws_sizing.get('deployment_recommendation', {})
        
        # Deployment Recommendation
        ax1.axis('off')
        ax1.text(0.05, 0.95, 'Recommended Deployment', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax1.transAxes)
        
        recommendation = deployment_rec.get('recommendation', 'unknown').upper()
        confidence = deployment_rec.get('confidence', 0) * 100
        
        if recommendation == 'RDS':
            rds_rec = aws_sizing.get('rds_recommendations', {})
            deploy_text = f"""
            Deployment: Amazon RDS Managed Service
            Instance: {rds_rec.get('primary_instance', 'N/A')}
            Storage: {rds_rec.get('storage_size_gb', 0):,.0f} GB ({rds_rec.get('storage_type', 'gp3').upper()})
            Multi-AZ: {'Yes' if rds_rec.get('multi_az', False) else 'No'}
            Monthly Cost: ${rds_rec.get('total_monthly_cost', 0):,.0f}
            AI Confidence: {confidence:.1f}%
            """
        else:
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            deploy_text = f"""
            Deployment: Amazon EC2 Self-Managed
            Instance: {ec2_rec.get('primary_instance', 'N/A')}
            Storage: {ec2_rec.get('storage_size_gb', 0):,.0f} GB ({ec2_rec.get('storage_type', 'gp3').upper()})
            EBS Optimized: {'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'}
            Monthly Cost: ${ec2_rec.get('total_monthly_cost', 0):,.0f}
            AI Confidence: {confidence:.1f}%
            """
        
        ax1.text(0.05, 0.75, deploy_text, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        
        # RDS vs EC2 Comparison
        rds_score = deployment_rec.get('rds_score', 0)
        ec2_score = deployment_rec.get('ec2_score', 0)
        
        comparison_data = {'RDS Score': rds_score, 'EC2 Score': ec2_score}
        self._create_bar_chart(ax2, comparison_data, 'Deployment Scoring', 'Score')
        
        # Reader/Writer Configuration
        reader_writer = aws_sizing.get('reader_writer_config', {})
        
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Instance Configuration', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        instance_text = f"""
        Writer Instances: {reader_writer.get('writers', 1)}
        Reader Instances: {reader_writer.get('readers', 0)}
        Total Instances: {reader_writer.get('total_instances', 1)}
        Read Capacity: {reader_writer.get('read_capacity_percent', 0):.1f}%
        Write Capacity: {reader_writer.get('write_capacity_percent', 100):.1f}%
        Recommended Read Split: {reader_writer.get('recommended_read_split', 0):.0f}%
        
        AI Reasoning:
        {reader_writer.get('reasoning', 'Standard configuration')[:100]}...
        """
        
        ax3.text(0.05, 0.75, instance_text, fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        # AI Complexity Factors
        ai_analysis = aws_sizing.get('ai_analysis', {})
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        
        self._create_gauge_chart(ax4, complexity_score * 10, 'AI Complexity Score', f'{complexity_score}/10')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_cost_analysis_page(self, pdf, analysis: Dict, config: Dict):
        """Create cost analysis page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Cost Analysis & Financial Projections', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        cost_analysis = analysis.get('cost_analysis', {})
        
        # Monthly Cost Breakdown (Pie Chart)
        cost_breakdown = {
            'Compute': cost_analysis.get('aws_compute_cost', 0),
            'Storage': cost_analysis.get('aws_storage_cost', 0),
            'Network': cost_analysis.get('network_cost', 0),
            'Agents': cost_analysis.get('agent_cost', 0),
            'Destination Storage': cost_analysis.get('destination_storage_cost', 0),
            'OS Licensing': cost_analysis.get('os_licensing_cost', 0),
            'Management': cost_analysis.get('management_cost', 0)
        }
        
        # Filter out zero values
        cost_breakdown = {k: v for k, v in cost_breakdown.items() if v > 0}
        
        self._create_pie_chart(ax1, cost_breakdown, 'Monthly Cost Breakdown')
        
        # Cost Projections
        monthly_cost = cost_analysis.get('total_monthly_cost', 0)
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        
        projections = {
            '1 Year': monthly_cost * 12 + one_time_cost,
            '2 Years': monthly_cost * 24 + one_time_cost,
            '3 Years': monthly_cost * 36 + one_time_cost,
            '5 Years': monthly_cost * 60 + one_time_cost
        }
        
        self._create_bar_chart(ax2, projections, 'Total Cost Projections', 'Cost ($)')
        
        # Savings Analysis
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Savings & ROI Analysis', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        savings_text = f"""
        Monthly Savings: ${cost_analysis.get('estimated_monthly_savings', 0):,.0f}
        Annual Savings: ${cost_analysis.get('estimated_monthly_savings', 0) * 12:,.0f}
        ROI Timeline: {cost_analysis.get('roi_months', 'TBD')} months
        Break-even Point: Year {cost_analysis.get('roi_months', 12) / 12:.1f}
        
        Destination Storage: {cost_analysis.get('destination_storage_type', 'S3')}
        Storage Cost: ${cost_analysis.get('destination_storage_cost', 0):,.0f}/month
        
        AI Cost Insights:
        - Optimization Factor: {cost_analysis.get('ai_cost_insights', {}).get('ai_optimization_factor', 0)*100:.1f}%
        - Complexity Multiplier: {cost_analysis.get('ai_cost_insights', {}).get('complexity_multiplier', 1.0):.2f}x
        - Additional Savings: {cost_analysis.get('ai_cost_insights', {}).get('potential_additional_savings', '0%')}
        """
        
        ax3.text(0.05, 0.75, savings_text, fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        # Cost Comparison
        current_cost_estimate = monthly_cost * 0.8  # Assume current is 20% higher
        cost_comparison = {
            'Current (Est.)': current_cost_estimate,
            'AWS Monthly': monthly_cost
        }
        
        self._create_bar_chart(ax4, cost_comparison, 'Cost Comparison', 'Monthly Cost ($)')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_risk_timeline_page(self, pdf, analysis: Dict, config: Dict):
        """Create risk assessment and timeline page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Risk Assessment & Project Timeline', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        ai_assessment = analysis.get('ai_overall_assessment', {})
        
        # Risk Assessment
        readiness_score = ai_assessment.get('migration_readiness_score', 0)
        success_prob = ai_assessment.get('success_probability', 0)
        risk_level = ai_assessment.get('risk_level', 'Medium')
        
        risk_data = {
            'Success Probability': success_prob,
            'Risk Mitigation': 100 - (100 - readiness_score) * 0.8
        }
        
        self._create_bar_chart(ax1, risk_data, 'Risk Assessment', 'Percentage (%)')
        
        # Timeline Visualization
        timeline = ai_assessment.get('timeline_recommendation', {})
        
        timeline_data = {
            'Planning': timeline.get('planning_phase_weeks', 2),
            'Testing': timeline.get('testing_phase_weeks', 3),
            'Migration': timeline.get('migration_window_hours', 24) / (7 * 24),  # Convert to weeks
            'Validation': 1
        }
        
        self._create_bar_chart(ax2, timeline_data, 'Project Timeline', 'Duration (Weeks)')
        
        # Risk Factors
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Key Risk Factors', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        risk_factors = ai_assessment.get('readiness_factors', [])
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        ai_analysis = aws_sizing.get('ai_analysis', {})
        risk_percentages = ai_analysis.get('risk_percentages', {})
        
        risk_text = "Identified Risk Factors:\n"
        for i, factor in enumerate(risk_factors[:4], 1):
            risk_text += f"{i}. {factor}\n"
        
        if risk_percentages:
            risk_text += "\nQuantified Risks:\n"
            for risk, percentage in list(risk_percentages.items())[:3]:
                risk_text += f"• {risk.replace('_', ' ').title()}: {percentage}%\n"
        
        ax3.text(0.05, 0.85, risk_text, fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        # Next Steps
        ax4.axis('off')
        ax4.text(0.05, 0.95, 'Recommended Next Steps', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax4.transAxes)
        
        next_steps = ai_assessment.get('recommended_next_steps', [])
        
        steps_text = ""
        for i, step in enumerate(next_steps, 1):
            steps_text += f"{i}. {step}\n"
        
        ax4.text(0.05, 0.85, steps_text, fontsize=9, transform=ax4.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_gauge_chart(self, ax, value, title, subtitle):
        """Create a gauge chart"""
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax.plot(theta, r, color='lightgray', linewidth=8)
        
        # Value arc
        value_theta = np.linspace(0, (value/100) * np.pi, int(value))
        value_r = np.ones_like(value_theta)
        
        color = '#e74c3c' if value < 50 else '#f39c12' if value < 75 else '#27ae60'
        ax.plot(value_theta, value_r, color=color, linewidth=8)
        
        # Add value text
        ax.text(0, -0.3, f'{value:.1f}', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0, -0.5, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0, -0.65, subtitle, ha='center', va='center', fontsize=10)
        
        ax.set_ylim(-0.8, 1.2)
        ax.set_xlim(-1.2, 1.2)
        ax.axis('off')
    
    def _create_radar_chart(self, ax, data, title):
        """Create a radar chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        # Number of variables
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        # Add values
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=self.report_style['accent_color'])
        ax.fill(angles, values, alpha=0.25, color=self.report_style['accent_color'])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True)
    
    def _create_bar_chart(self, ax, data, title, ylabel):
        """Create a bar chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        bars = ax.bar(labels, values, color=self.report_style['accent_color'], alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis for currency if needed
        if '$' in ylabel or 'Cost' in ylabel:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _create_pie_chart(self, ax, data, title):
        """Create a pie chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        # Filter out zero values
        filtered_data = [(label, value) for label, value in zip(labels, values) if value > 0]
        if filtered_data:
            labels, values = zip(*filtered_data)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        
        ax.set_title(title, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

# Network path diagram function
def create_network_path_diagram(network_perf: Dict) -> go.Figure:
    """Create an interactive network path diagram"""
    
    segments = network_perf.get('segments', [])
    if not segments:
        return go.Figure()
    
    # Create network diagram using plotly
    fig = go.Figure()
    
    # Define positions for network nodes
    num_segments = len(segments)
    x_positions = [i * 100 for i in range(num_segments + 1)]
    y_positions = [50] * (num_segments + 1)
    
    # Add network segments as lines
    for i, segment in enumerate(segments):
        # Calculate line properties based on performance
        line_width = max(2, min(10, segment['effective_bandwidth_mbps'] / 200))
        
        # Color based on performance (green = good, yellow = ok, red = poor)
        reliability = segment['reliability']
        if reliability > 0.999:
            line_color = '#27ae60'  # Green
        elif reliability > 0.995:
            line_color = '#f39c12'  # Orange
        else:
            line_color = '#e74c3c'  # Red
        
        # Add line for network segment
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i+1]],
            y=[y_positions[i], y_positions[i+1]],
            mode='lines+markers',
            line=dict(
                width=line_width,
                color=line_color
            ),
            marker=dict(size=15, color='#2c3e50', symbol='square'),
            name=segment['name'],
            hovertemplate=f"""
            <b>{segment['name']}</b><br>
            Type: {segment['connection_type'].replace('_', ' ').title()}<br>
            Bandwidth: {segment['effective_bandwidth_mbps']:.0f} Mbps<br>
            Latency: {segment['effective_latency_ms']:.1f} ms<br>
            Reliability: {segment['reliability']*100:.3f}%<br>
            Cost Factor: {segment['cost_factor']:.1f}x<br>
            AI Optimization: {segment.get('ai_optimization_potential', 0)*100:.1f}%<br>
            <extra></extra>
            """
        ))
        
        # Add bandwidth and latency annotations
        mid_x = (x_positions[i] + x_positions[i+1]) / 2
        mid_y = (y_positions[i] + y_positions[i+1]) / 2 + 20
        
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=f"<b>{segment['effective_bandwidth_mbps']:.0f} Mbps</b><br>{segment['effective_latency_ms']:.1f} ms",
            showarrow=False,
            font=dict(size=10, color='#2c3e50'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#bdc3c7',
            borderwidth=1,
            borderpad=4
        )
        
        # Add connection type label
        fig.add_annotation(
            x=mid_x,
            y=mid_y - 25,
            text=f"<i>{segment['connection_type'].replace('_', ' ').title()}</i>",
            showarrow=False,
            font=dict(size=8, color='#7f8c8d'),
            bgcolor='rgba(248,249,250,0.8)',
            bordercolor='#dee2e6',
            borderwidth=1,
            borderpad=2
        )
    
    # Add source and destination nodes
    fig.add_trace(go.Scatter(
        x=[x_positions[0]],
        y=[y_positions[0]],
        mode='markers+text',
        marker=dict(size=25, color='#27ae60', symbol='circle'),
        text=['SOURCE'],
        textposition='bottom center',
        name='Source System',
        hovertemplate="<b>Source System</b><br>On-Premises Database<extra></extra>"
    ))
    
    destination_storage = network_perf.get('destination_storage', 'S3')
    destination_color = {
        'S3': '#3498db',
        'FSx_Windows': '#e74c3c', 
        'FSx_Lustre': '#9b59b6'
    }.get(destination_storage, '#3498db')
    
    fig.add_trace(go.Scatter(
        x=[x_positions[-1]],
        y=[y_positions[-1]],
        mode='markers+text',
        marker=dict(size=25, color=destination_color, symbol='circle'),
        text=[destination_storage],
        textposition='bottom center',
        name=f'AWS {destination_storage}',
        hovertemplate=f"<b>AWS Destination</b><br>{destination_storage} Storage Service<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Network Migration Path: {network_perf.get('path_name', 'Unknown')} → {destination_storage}",
            font=dict(size=16, color='#2c3e50'),
            x=0.5
        ),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[-20, max(x_positions) + 20]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[0, 100]
        ),
        showlegend=False,
        height=350,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Enhanced rendering functions

def render_enhanced_header():
    """Enhanced header with professional styling"""
    
    # Initialize managers to check status
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
    ai_status = "🟢" if ai_manager.connected else "🔴"
    aws_status = "🟢" if aws_api.connected else "🔴"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration • Agent Scaling Optimization • FSx Destination Analysis
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
            Comprehensive Network Path Analysis • OS Performance Optimization • Enterprise-Ready Migration Planning • Multi-Agent Coordination • S3/FSx Comparisons
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
            <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
            <span style="margin-right: 20px;">🟢 Network Intelligence Engine</span>
            <span style="margin-right: 20px;">🟢 Agent Scaling Optimizer</span>
            <span>🟢 FSx Destination Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_api_status_sidebar():
    """Enhanced API status sidebar"""
    
    st.sidebar.markdown("### 🔌 System Status")
    
    # Check API status
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
    # Anthropic AI Status
    ai_status_class = "status-online" if ai_manager.connected else "status-offline"
    ai_status_text = "Connected" if ai_manager.connected else "Disconnected"
    
    st.sidebar.markdown(f"""
    <div class="api-status-card">
        <span class="status-indicator {ai_status_class}"></span>
        <strong>Anthropic Claude AI:</strong> {ai_status_text}
        {f"<br><small>Error: {ai_manager.error_message[:50]}...</small>" if ai_manager.error_message else ""}
    </div>
    """, unsafe_allow_html=True)
    
    # AWS API Status
    aws_status_class = "status-online" if aws_api.connected else "status-warning"
    aws_status_text = "Connected" if aws_api.connected else "Using Fallback Data"
    
    st.sidebar.markdown(f"""
    <div class="api-status-card">
        <span class="status-indicator {aws_status_class}"></span>
        <strong>AWS Pricing API:</strong> {aws_status_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration instructions
    if not ai_manager.connected or not aws_api.connected:
        st.sidebar.markdown("### ⚙️ Configuration")
        
        if not ai_manager.connected:
            st.sidebar.info("💡 Add ANTHROPIC_API_KEY to Streamlit secrets for enhanced AI analysis")
        
        if not aws_api.connected:
            st.sidebar.info("💡 Configure AWS credentials for real-time pricing data")

def render_enhanced_sidebar_controls():
    """Enhanced sidebar with AI-powered recommendations, agent scaling, and FSx destination selection"""
    
    st.sidebar.header("🤖 AI-Powered Migration Configuration v3.0 with FSx Analysis")
    
    # Render API status
    render_api_status_sidebar()
    
    st.sidebar.markdown("---")
    
    # Operating System Selection with AI insights
    st.sidebar.subheader("💻 Operating System (AI-Enhanced)")
    operating_system = st.sidebar.selectbox(
        "OS Selection",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3,
        format_func=lambda x: {
            'windows_server_2019': '🔵 Windows Server 2019',
            'windows_server_2022': '🔵 Windows Server 2022 (Latest)',
            'rhel_8': '🔴 Red Hat Enterprise Linux 8',
            'rhel_9': '🔴 Red Hat Enterprise Linux 9 (Latest)',
            'ubuntu_20_04': '🟠 Ubuntu Server 20.04 LTS',
            'ubuntu_22_04': '🟠 Ubuntu Server 22.04 LTS (Latest)'
        }[x],
        help="AI analyzes OS performance characteristics and migration impact"
    )
    
    # Show AI OS insights
    os_manager = OSPerformanceManager()
    os_config = os_manager.operating_systems[operating_system]
    
    with st.sidebar.expander("🤖 AI OS Insights"):
        st.markdown(f"**Strengths:** {', '.join(os_config['ai_insights']['strengths'][:2])}")
        st.markdown(f"**Key Consideration:** {os_config['ai_insights']['weaknesses'][0]}")
    
    # Platform Configuration
    st.sidebar.subheader("🖥️ Server Platform")
    server_type = st.sidebar.selectbox(
        "Platform Type",
        ["physical", "vmware"],
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine",
        help="Physical vs Virtual performance analysis with AI optimization"
    )
    
    # Hardware Configuration with AI recommendations
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256, 512], index=2, 
                                 help="AI calculates optimal memory for database workload")
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32, 48, 64], index=2,
                                   help="AI analyzes CPU requirements for migration performance")
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
    # Enhanced Performance Metrics
    st.sidebar.subheader("📊 Current Performance Metrics")
    current_storage_gb = st.sidebar.number_input("Current Storage (GB)", 
                                                min_value=100, max_value=500000, value=2000, step=100,
                                                help="Current storage capacity in use")
    peak_iops = st.sidebar.number_input("Peak IOPS", 
                                       min_value=100, max_value=1000000, value=10000, step=500,
                                       help="Maximum IOPS observed during peak usage")
    max_throughput_mbps = st.sidebar.number_input("Max Throughput (MB/s)", 
                                                 min_value=10, max_value=10000, value=500, step=50,
                                                 help="Maximum storage throughput observed")
    anticipated_max_memory_gb = st.sidebar.number_input("Anticipated Max Memory (GB)", 
                                                       min_value=4, max_value=1024, value=64, step=8,
                                                       help="Maximum memory usage anticipated for workload")
    anticipated_max_cpu_cores = st.sidebar.number_input("Anticipated Max CPU Cores", 
                                                       min_value=1, max_value=128, value=16, step=2,
                                                       help="Maximum CPU cores anticipated for workload")
    
    # Network Interface with AI insights
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"],
        index=3,
        format_func=lambda x: {
            'gigabit_copper': '🔶 1Gbps Copper',
            'gigabit_fiber': '🟡 1Gbps Fiber',
            '10g_copper': '🔵 10Gbps Copper',
            '10g_fiber': '🟢 10Gbps Fiber',
            '25g_fiber': '🟣 25Gbps Fiber',
            '40g_fiber': '🔴 40Gbps Fiber'
        }[x],
        help="AI analyzes network impact on migration throughput"
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000, 
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    # Migration Configuration with AI analysis
    st.sidebar.subheader("🔄 Migration Setup (AI-Optimized)")
    
    # Source and Target Databases
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x],
        help="AI determines migration complexity based on source engine"
    )
    
    database_engine = st.sidebar.selectbox(
        "Target Database (AWS)",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '☁️ RDS MySQL', 'postgresql': '☁️ RDS PostgreSQL', 'oracle': '☁️ RDS Oracle',
            'sqlserver': '☁️ RDS SQL Server', 'mongodb': '☁️ DocumentDB'
        }[x],
        help="AI recommends optimal AWS database service"
    )
    
    # Show migration type indicator
    is_homogeneous = source_database_engine == database_engine
    migration_type_indicator = "🟢 Homogeneous" if is_homogeneous else "🟡 Heterogeneous"
    st.sidebar.info(f"**Migration Type:** {migration_type_indicator}")
    
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100,
                                              help="AI calculates migration time and resource requirements")
    
    # Migration Parameters
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60,
                                                        help="AI optimizes migration strategy for downtime window")
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"],
                                                   help="AI adjusts AWS sizing recommendations")
    
    # NEW: Destination Storage Selection
    st.sidebar.subheader("🗄️ Destination Storage (Enhanced Analysis)")
    destination_storage_type = st.sidebar.selectbox(
        "AWS Destination Storage",
        ["S3", "FSx_Windows", "FSx_Lustre"],
        format_func=lambda x: {
            'S3': '☁️ Amazon S3 (Standard)',
            'FSx_Windows': '🪟 Amazon FSx for Windows File Server',
            'FSx_Lustre': '⚡ Amazon FSx for Lustre (High Performance)'
        }[x],
        help="AI analyzes performance, cost, and complexity for each destination type"
    )
    
    # Show destination storage insights
    with st.sidebar.expander("🎯 Storage Destination Insights"):
        if destination_storage_type == "S3":
            st.markdown("""
            **S3 Advantages:**
            • Cost-effective and scalable
            • Simple migration integration
            • Excellent durability (99.999999999%)
            
            **Best For:** General-purpose storage, cost optimization
            """)
        elif destination_storage_type == "FSx_Windows":
            st.markdown("""
            **FSx Windows Advantages:**
            • Native Windows file system features
            • Better performance than S3
            • Active Directory integration
            
            **Best For:** Windows-based applications, file shares
            """)
        elif destination_storage_type == "FSx_Lustre":
            st.markdown("""
            **FSx Lustre Advantages:**
            • Extremely high performance (sub-ms latency)
            • Parallel processing optimized
            • Perfect for HPC and analytics
            
            **Best For:** High-performance computing, ML workloads
            """)
    
    # Environment
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"],
                                     help="AI adjusts reliability and performance requirements")
    
    # Enhanced Agent Sizing Section with AI recommendations
    st.sidebar.subheader("🤖 Migration Agent Configuration (AI-Optimized)")
    
    # Determine migration type for tool selection
    primary_tool = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.success(f"**Primary Tool:** AWS {primary_tool}")
    
    # Agent Count Configuration
    st.sidebar.markdown("**📊 Agent Scaling Configuration:**")
    
    number_of_agents = st.sidebar.number_input(
        "Number of Migration Agents",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Configure the number of agents for parallel migration processing"
    )
    
    # Show agent scaling insights
    if number_of_agents == 1:
        st.sidebar.info("💡 Single agent - simple but may limit throughput")
    elif number_of_agents <= 3:
        st.sidebar.success("✅ Optimal range for most workloads")
    elif number_of_agents <= 5:
        st.sidebar.warning("⚠️ High agent count - ensure proper coordination")
    else:
        st.sidebar.error("🔴 Very high agent count - diminishing returns likely")
    
    if is_homogeneous:
        datasync_agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': '📦 Small (t3.medium) - 250 Mbps/agent',
                'medium': '📦 Medium (c5.large) - 500 Mbps/agent',
                'large': '📦 Large (c5.xlarge) - 1000 Mbps/agent',
                'xlarge': '📦 XLarge (c5.2xlarge) - 2000 Mbps/agent'
            }[x],
            help=f"AI recommends optimal agent size for {number_of_agents} agents"
        )
        dms_agent_size = None
    else:
        dms_agent_size = st.sidebar.selectbox(
            "DMS Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: {
                'small': '🔄 Small (t3.medium) - 200 Mbps/agent',
                'medium': '🔄 Medium (c5.large) - 400 Mbps/agent',
                'large': '🔄 Large (c5.xlarge) - 800 Mbps/agent',
                'xlarge': '🔄 XLarge (c5.2xlarge) - 1500 Mbps/agent',
                'xxlarge': '🔄 XXLarge (c5.4xlarge) - 2500 Mbps/agent'
            }[x],
            help=f"AI recommends optimal instance size for {number_of_agents} agents"
        )
        datasync_agent_size = None
    
    # Show estimated throughput with current configuration including destination storage impact
    agent_manager = EnhancedAgentSizingManager()
    if is_homogeneous:
        test_config = agent_manager.calculate_agent_configuration('datasync', datasync_agent_size, number_of_agents, destination_storage_type)
    else:
        test_config = agent_manager.calculate_agent_configuration('dms', dms_agent_size, number_of_agents, destination_storage_type)
    
    st.sidebar.markdown(f"""
    <div class="agent-scaling-card">
        <h4>🚀 Current Configuration Impact</h4>
        <p><strong>Total Throughput:</strong> {test_config['total_max_throughput_mbps']:,.0f} Mbps</p>
        <p><strong>Scaling Efficiency:</strong> {test_config['scaling_efficiency']*100:.1f}%</p>
        <p><strong>Storage Multiplier:</strong> {test_config['storage_performance_multiplier']:.1f}x</p>
        <p><strong>Monthly Cost:</strong> ${test_config['total_monthly_cost']:,.0f}</p>
        <p><strong>Config Rating:</strong> {test_config['optimal_configuration']['efficiency_score']:.0f}/100</p>
        <p><strong>Destination:</strong> {destination_storage_type}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Configuration Section
    st.sidebar.subheader("🧠 AI Configuration")
    
    enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=True,
                                           help="Use Anthropic AI for intelligent recommendations")
    
    if enable_ai_analysis:
        ai_analysis_depth = st.sidebar.selectbox(
            "AI Analysis Depth",
            ["standard", "comprehensive"],
            help="Comprehensive analysis provides more detailed AI insights"
        )
    else:
        ai_analysis_depth = "standard"
    
    # Real-time AWS Pricing
    use_realtime_pricing = st.sidebar.checkbox("Real-time AWS Pricing", value=True,
                                             help="Fetch current AWS pricing via API")
    
    if st.sidebar.button("🔄 Refresh AI Analysis", type="primary"):
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
        'destination_storage_type': destination_storage_type,
        'environment': environment,
        'datasync_agent_size': datasync_agent_size,
        'dms_agent_size': dms_agent_size,
        'number_of_agents': number_of_agents,
        'enable_ai_analysis': enable_ai_analysis,
        'ai_analysis_depth': ai_analysis_depth,
        'use_realtime_pricing': use_realtime_pricing,
        'current_storage_gb': current_storage_gb,
        'peak_iops': peak_iops,
        'max_throughput_mbps': max_throughput_mbps,
        'anticipated_max_memory_gb': anticipated_max_memory_gb,
        'anticipated_max_cpu_cores': anticipated_max_cpu_cores
    }

def render_fsx_destination_comparison_tab(analysis: Dict, config: Dict):
    """Render FSx destination comparison analysis tab"""
    st.subheader("🗄️ FSx Destination Storage Comparison & Performance Analysis")
    
    fsx_comparisons = analysis.get('fsx_comparisons', {})
    current_destination = config.get('destination_storage_type', 'S3')
    
    if not fsx_comparisons:
        st.error("FSx comparison data not available. Please run the analysis first.")
        return
    
    # Destination Overview Cards
    st.markdown("**📊 Destination Storage Overview:**")
    
    col1, col2, col3 = st.columns(3)
    
    destinations = list(fsx_comparisons.keys())
    for i, (col, destination) in enumerate(zip([col1, col2, col3], destinations)):
        comparison = fsx_comparisons.get(destination, {})
        
        # Determine if this is the current selection
        is_current = destination == current_destination
        card_style = "storage-destination-card" if is_current else "detailed-analysis-section"
        
        with col:
            st.markdown(f"""
            <div class="{card_style}">
                <h4>{'🎯 ' if is_current else ''}{destination} {'(Selected)' if is_current else ''}</h4>
                <p><strong>Performance:</strong> {comparison.get('performance_rating', 'Unknown')}</p>
                <p><strong>Cost Rating:</strong> {comparison.get('cost_rating', 'Unknown')}</p>
                <p><strong>Complexity:</strong> {comparison.get('complexity_rating', 'Unknown')}</p>
                <p><strong>Migration Time:</strong> {comparison.get('estimated_migration_time_hours', 0):.1f} hours</p>
                <p><strong>Monthly Storage Cost:</strong> ${comparison.get('estimated_monthly_storage_cost', 0):,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance Comparison Charts
    st.markdown("**📈 Performance & Cost Comparison:**")
    
    # Create comparison visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Migration Time Comparison
        migration_times = {dest: fsx_comparisons[dest].get('estimated_migration_time_hours', 0) 
                          for dest in destinations}
        
        fig_time = px.bar(
            x=list(migration_times.keys()),
            y=list(migration_times.values()),
            title="Migration Time Comparison",
            labels={'x': 'Destination Storage', 'y': 'Migration Time (Hours)'},
            color=list(migration_times.values()),
            color_continuous_scale='RdYlGn_r'
        )
        
        # Highlight current selection
        colors = ['red' if dest == current_destination else 'lightblue' for dest in destinations]
        fig_time.update_traces(marker_color=colors)
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Cost Comparison
        storage_costs = {dest: fsx_comparisons[dest].get('estimated_monthly_storage_cost', 0) 
                        for dest in destinations}
        
        fig_cost = px.bar(
            x=list(storage_costs.keys()),
            y=list(storage_costs.values()),
            title="Monthly Storage Cost Comparison",
            labels={'x': 'Destination Storage', 'y': 'Monthly Cost ($)'},
            color=list(storage_costs.values()),
            color_continuous_scale='RdYlBu_r'
        )
        
        # Highlight current selection
        colors = ['red' if dest == current_destination else 'lightgreen' for dest in destinations]
        fig_cost.update_traces(marker_color=colors)
        
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Detailed Analysis per Destination
    st.markdown("**🔍 Detailed Destination Analysis:**")
    
    destination_tabs = st.tabs([f"{dest}" for dest in destinations])
    
    for tab, destination in zip(destination_tabs, destinations):
        with tab:
            comparison = fsx_comparisons.get(destination, {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="fsx-comparison-card">
                    <h4>{destination} Performance Metrics</h4>
                    <p><strong>Migration Time:</strong> {comparison.get('estimated_migration_time_hours', 0):.1f} hours</p>
                    <p><strong>Throughput:</strong> {comparison.get('migration_throughput_mbps', 0):,.0f} Mbps</p>
                    <p><strong>Performance Rating:</strong> {comparison.get('performance_rating', 'Unknown')}</p>
                    <p><strong>Cost Rating:</strong> {comparison.get('cost_rating', 'Unknown')}</p>
                    <p><strong>Complexity Rating:</strong> {comparison.get('complexity_rating', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="fsx-comparison-card">
                    <h4>{destination} Cost Analysis</h4>
                    <p><strong>Monthly Storage:</strong> ${comparison.get('estimated_monthly_storage_cost', 0):,.0f}</p>
                    <p><strong>Agent Configuration:</strong> {comparison.get('agent_configuration', {}).get('number_of_agents', 1)} agents</p>
                    <p><strong>Agent Monthly Cost:</strong> ${comparison.get('agent_configuration', {}).get('total_monthly_cost', 0):,.0f}</p>
                    <p><strong>Storage Multiplier:</strong> {comparison.get('agent_configuration', {}).get('storage_performance_multiplier', 1.0):.1f}x</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations for this destination
            recommendations = comparison.get('recommendations', [])
            if recommendations:
                st.markdown(f"**💡 {destination} Recommendations:**")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
            
            # Network performance for this destination
            network_perf = comparison.get('network_performance', {})
            if network_perf:
                st.markdown(f"**🌐 {destination} Network Performance:**")
                
                network_col1, network_col2 = st.columns(2)
                
                with network_col1:
                    st.metric("Network Quality", f"{network_perf.get('network_quality_score', 0):.1f}/100")
                    st.metric("AI Enhanced Quality", f"{network_perf.get('ai_enhanced_quality_score', 0):.1f}/100")
                
                with network_col2:
                    st.metric("Effective Bandwidth", f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps")
                    st.metric("Total Latency", f"{network_perf.get('total_latency_ms', 0):.1f} ms")
    
    # Summary Recommendations
    st.markdown("**🎯 Overall Destination Recommendations:**")
    
    # Find the best option for different criteria
    best_performance = max(destinations, key=lambda x: fsx_comparisons[x].get('estimated_migration_time_hours', float('inf')), default='S3')
    best_cost = min(destinations, key=lambda x: fsx_comparisons[x].get('estimated_monthly_storage_cost', float('inf')), default='S3')
    
    # Invert for performance (lower time is better)
    best_performance = min(destinations, key=lambda x: fsx_comparisons[x].get('estimated_migration_time_hours', float('inf')), default='S3')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="performance-comparison-card">
            <h4>🚀 Best Performance</h4>
            <p><strong>Winner:</strong> {best_performance}</p>
            <p><strong>Migration Time:</strong> {fsx_comparisons[best_performance].get('estimated_migration_time_hours', 0):.1f} hours</p>
            <p><strong>Throughput:</strong> {fsx_comparisons[best_performance].get('migration_throughput_mbps', 0):,.0f} Mbps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="performance-comparison-card">
            <h4>💰 Best Cost Efficiency</h4>
            <p><strong>Winner:</strong> {best_cost}</p>
            <p><strong>Monthly Cost:</strong> ${fsx_comparisons[best_cost].get('estimated_monthly_storage_cost', 0):,.0f}</p>
            <p><strong>Cost Rating:</strong> {fsx_comparisons[best_cost].get('cost_rating', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        current_comparison = fsx_comparisons.get(current_destination, {})
        st.markdown(f"""
        <div class="performance-comparison-card">
            <h4>🎯 Current Selection</h4>
            <p><strong>Choice:</strong> {current_destination}</p>
            <p><strong>Performance:</strong> {current_comparison.get('performance_rating', 'Unknown')}</p>
            <p><strong>Complexity:</strong> {current_comparison.get('complexity_rating', 'Unknown')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Decision Matrix
    st.markdown("**📊 Decision Matrix:**")
    
    # Create decision matrix data
    matrix_data = []
    for dest in destinations:
        comp = fsx_comparisons[dest]
        matrix_data.append({
            'Destination': dest,
            'Performance Rating': comp.get('performance_rating', 'Unknown'),
            'Cost Rating': comp.get('cost_rating', 'Unknown'),
            'Complexity Rating': comp.get('complexity_rating', 'Unknown'),
            'Migration Time (Hours)': f"{comp.get('estimated_migration_time_hours', 0):.1f}",
            'Monthly Cost ($)': f"${comp.get('estimated_monthly_storage_cost', 0):,.0f}",
            'Current Selection': '✅' if dest == current_destination else ''
        })
    
    df_matrix = pd.DataFrame(matrix_data)
    st.dataframe(df_matrix, use_container_width=True)

def render_ai_insights_tab_enhanced(analysis: Dict, config: Dict):
    """Enhanced AI insights tab with detailed analysis including agent scaling and FSx destinations"""
    st.subheader("🧠 Comprehensive AI-Powered Migration Analysis")
    
    ai_assessment = analysis.get('ai_overall_assessment', {})
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    ai_analysis = aws_sizing.get('ai_analysis', {})
    agent_analysis = analysis.get('agent_analysis', {})
    
    # Enhanced Migration Readiness Dashboard with agent scaling and destination storage metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        readiness_score = ai_assessment.get('migration_readiness_score', 0)
        success_prob = ai_assessment.get('success_probability', 0)
        st.metric(
            "🎯 Migration Readiness", 
            f"{readiness_score:.0f}/100",
            delta=f"Success Rate: {success_prob:.0f}%"
        )
    
    with col2:
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        risk_level = ai_assessment.get('risk_level', 'Unknown')
        st.metric(
            "🔄 Complexity Analysis",
            f"{complexity_score:.1f}/10",
            delta=f"Risk: {risk_level}"
        )
    
    with col3:
        confidence = ai_analysis.get('confidence_level', 'medium')
        ai_confidence = ai_assessment.get('ai_confidence', 0.5)
        st.metric(
            "🤖 AI Confidence",
            f"{ai_confidence*100:.1f}%",
            delta=confidence.title()
        )
    
    with col4:
        timeline = ai_assessment.get('timeline_recommendation', {})
        total_weeks = timeline.get('total_project_weeks', 6)
        approach = timeline.get('recommended_approach', 'direct')
        st.metric(
            "📅 Project Timeline",
            f"{total_weeks:.0f} weeks",
            delta=approach.replace('_', ' ').title()
        )
    
    with col5:
        resource_allocation = ai_analysis.get('resource_allocation', {})
        team_size = resource_allocation.get('migration_team_size', 3)
        specialists = resource_allocation.get('aws_specialists_needed', 1)
        st.metric(
            "👥 Team Requirements",
            f"{team_size:.0f} members",
            delta=f"{specialists} AWS specialists"
        )
    
    # Agent Scaling Performance Metrics
    st.markdown("**🤖 Agent Scaling Performance Analysis:**")
    
    agent_col1, agent_col2, agent_col3, agent_col4, agent_col5 = st.columns(5)
    
    with agent_col1:
        num_agents = config.get('number_of_agents', 1)
        optimal_agents = agent_analysis.get('optimal_recommendations', {}).get('optimal_configuration', {}).get('configuration', {}).get('number_of_agents', num_agents)
        st.metric(
            "🔧 Agent Configuration",
            f"{num_agents} agents",
            delta=f"Optimal: {optimal_agents}"
        )
    
    with agent_col2:
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0)
        management_overhead = agent_analysis.get('management_overhead', 1.0)
        st.metric(
            "⚡ Scaling Efficiency",
            f"{scaling_efficiency*100:.1f}%",
            delta=f"Overhead: {(management_overhead-1)*100:.1f}%"
        )
    
    with agent_col3:
        total_throughput = agent_analysis.get('total_effective_throughput', 0)
        per_agent_throughput = total_throughput / num_agents if num_agents > 0 else 0
        st.metric(
            "🚀 Total Throughput",
            f"{total_throughput:,.0f} Mbps",
            delta=f"{per_agent_throughput:.0f} Mbps/agent"
        )
    
    with agent_col4:
        bottleneck = agent_analysis.get('bottleneck', 'unknown')
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        st.metric(
            "🔍 Bottleneck Analysis",
            bottleneck.title(),
            delta=f"Severity: {bottleneck_severity.title()}"
        )
    
    with agent_col5:
        agent_monthly_cost = agent_analysis.get('monthly_cost', 0)
        cost_per_agent = agent_monthly_cost / num_agents if num_agents > 0 else 0
        st.metric(
            "💰 Agent Costs",
            f"${agent_monthly_cost:,.0f}/mo",
            delta=f"${cost_per_agent:.0f}/agent"
        )
    
    # Destination Storage Impact Analysis
    st.markdown("**🗄️ Destination Storage Impact Analysis:**")
    
    dest_col1, dest_col2, dest_col3, dest_col4, dest_col5 = st.columns(5)
    
    destination_storage = config.get('destination_storage_type', 'S3')
    storage_impact = ai_assessment.get('destination_storage_impact', {})
    
    with dest_col1:
        st.metric(
            "🎯 Current Destination",
            destination_storage,
            delta=f"Type: {storage_impact.get('storage_type', 'Unknown')}"
        )
    
    with dest_col2:
        performance_bonus = storage_impact.get('performance_bonus', 0)
        storage_multiplier = storage_impact.get('storage_performance_multiplier', 1.0)
        st.metric(
            "⚡ Performance Multiplier",
            f"{storage_multiplier:.1f}x",
            delta=f"Bonus: +{performance_bonus}%"
        )
    
    with dest_col3:
        # Get destination storage cost from cost analysis
        cost_analysis = analysis.get('cost_analysis', {})
        dest_storage_cost = cost_analysis.get('destination_storage_cost', 0)
        st.metric(
            "💰 Storage Cost",
            f"${dest_storage_cost:,.0f}/mo",
            delta=f"For {config.get('database_size_gb', 0):,} GB"
        )
    
    with dest_col4:
        # Calculate relative cost compared to S3
        fsx_comparisons = analysis.get('fsx_comparisons', {})
        s3_cost = fsx_comparisons.get('S3', {}).get('estimated_monthly_storage_cost', 0)
        current_cost = fsx_comparisons.get(destination_storage, {}).get('estimated_monthly_storage_cost', 0)
        cost_difference = ((current_cost - s3_cost) / s3_cost * 100) if s3_cost > 0 else 0
        st.metric(
            "📊 Cost vs S3",
            f"{cost_difference:+.0f}%",
            delta=f"${current_cost - s3_cost:+,.0f}/mo"
        )
    
    with dest_col5:
        # Get migration time benefit
        s3_time = fsx_comparisons.get('S3', {}).get('estimated_migration_time_hours', 0)
        current_time = fsx_comparisons.get(destination_storage, {}).get('estimated_migration_time_hours', 0)
        time_difference = ((s3_time - current_time) / s3_time * 100) if s3_time > 0 else 0
        st.metric(
            "⏱️ Time vs S3",
            f"{time_difference:+.0f}%",
            delta=f"{current_time - s3_time:+.1f} hours"
        )
    
    # Performance Metrics Overview
    st.markdown("**📊 Current Performance Analysis:**")
    
    perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
    
    with perf_col1:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>💾 Current Storage</h5>
            <p><strong>{config.get('current_storage_gb', 0):,} GB</strong></p>
            <p>Database: {config.get('database_size_gb', 0):,} GB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>⚡ Peak IOPS</h5>
            <p><strong>{config.get('peak_iops', 0):,}</strong></p>
            <p>Max observed performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>🚀 Max Throughput</h5>
            <p><strong>{config.get('max_throughput_mbps', 0)} MB/s</strong></p>
            <p>Storage throughput peak</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col4:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>🧠 Anticipated Memory</h5>
            <p><strong>{config.get('anticipated_max_memory_gb', 0)} GB</strong></p>
            <p>Current: {config.get('ram_gb', 0)} GB</p>
        </div>
        """, unsafe_allow_html=True)
    
    with perf_col5:
        st.markdown(f"""
        <div class="compact-metric">
            <h5>⚙️ Anticipated CPU</h5>
            <p><strong>{config.get('anticipated_max_cpu_cores', 0)} cores</strong></p>
            <p>Current: {config.get('cpu_cores', 0)} cores</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Agent Scaling Optimization Analysis
    st.markdown("**🎯 Agent Scaling Optimization Analysis:**")
    
    agent_optimization = agent_analysis.get('ai_optimization', {})
    optimal_recommendations = agent_analysis.get('optimal_recommendations', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="agent-scaling-card">
            <h4>🔧 Current Agent Configuration Analysis</h4>
            <p><strong>Configuration Efficiency:</strong> {agent_optimization.get('current_efficiency', 0):.1f}%</p>
            <p><strong>Optimization Potential:</strong> {agent_optimization.get('optimization_potential_percent', 0):.1f}%</p>
            <p><strong>Bottleneck Type:</strong> {agent_optimization.get('bottleneck_type', 'Unknown').title()}</p>
            <p><strong>Scaling Assessment:</strong> {agent_optimization.get('scaling_assessment', {}).get('cost_efficiency', 'Unknown')}</p>
            <p><strong>Estimated Improvement:</strong> {agent_optimization.get('estimated_improvement', 'N/A')}</p>
            <p><strong>Destination Impact:</strong> {agent_optimization.get('destination_storage_impact', 'Standard performance')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if optimal_recommendations.get('optimal_configuration'):
            optimal_config = optimal_recommendations['optimal_configuration']
            st.markdown(f"""
            <div class="agent-scaling-card">
                <h4>🎯 AI-Recommended Optimal Configuration</h4>
                <p><strong>Optimal Agents:</strong> {optimal_config['configuration']['number_of_agents']}</p>
                <p><strong>Agent Size:</strong> {optimal_config['configuration']['agent_size'].title()}</p>
                <p><strong>Destination:</strong> {optimal_config['configuration']['destination_storage']}</p>
                <p><strong>Total Throughput:</strong> {optimal_config['total_throughput_mbps']:,.0f} Mbps</p>
                <p><strong>Monthly Cost:</strong> ${optimal_config['total_cost_per_hour'] * 24 * 30:,.0f}</p>
                <p><strong>Overall Score:</strong> {optimal_config['overall_score']:.1f}/100</p>
                <p><strong>Storage Multiplier:</strong> {optimal_config.get('storage_performance_multiplier', 1.0):.1f}x</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agent-scaling-card">
                <h4>🎯 AI Analysis Status</h4>
                <p>Optimal configuration analysis not available</p>
                <p>Current configuration efficiency: {agent_optimization.get('current_efficiency', 0):.1f}%</p>
                <p>Destination storage: {destination_storage}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Agent Scaling Recommendations
    agent_recommendations = agent_optimization.get('recommendations', [])
    if agent_recommendations:
        st.markdown("**🤖 AI Agent Scaling Recommendations:**")
        
        for i, recommendation in enumerate(agent_recommendations, 1):
            priority = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            impact = "Significant" if i <= 2 else "Moderate" if i <= 4 else "Minor"
            
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h5>Recommendation {i}: {recommendation}</h5>
                <p><strong>Priority:</strong> {priority}</p>
                <p><strong>Expected Impact:</strong> {impact}</p>
                <p><strong>Implementation:</strong> {"Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"}</p>
                <p><strong>Destination Relevance:</strong> {"High" if destination_storage in recommendation else "Standard"}</p>
            </div>
            """)
    
    # Destination Storage Analysis
    st.markdown("**🗄️ Destination Storage Analysis:**")
    
    dest_ai_analysis = ai_analysis.get('destination_storage_impact', {})
    if dest_ai_analysis:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="ai-insight-card">
                <h4>📊 {destination_storage} Performance Analysis</h4>
                <p><strong>Performance Impact:</strong> {dest_ai_analysis.get('performance_impact', {}).get('performance_rating', 'Unknown')}</p>
                <p><strong>Throughput Multiplier:</strong> {dest_ai_analysis.get('performance_impact', {}).get('throughput_multiplier', 1.0):.1f}x</p>
                <p><strong>Latency Impact:</strong> {dest_ai_analysis.get('performance_impact', {}).get('latency_impact', 1.0):.1f}x</p>
                <p><strong>IOPS Capability:</strong> {dest_ai_analysis.get('performance_impact', {}).get('iops_capability', 'Standard')}</p>
                <p><strong>Complexity Factor:</strong> {dest_ai_analysis.get('complexity_factor', 1.0):.1f}x</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            recommendations_for_storage = dest_ai_analysis.get('recommended_for', [])
            st.markdown(f"""
            <div class="ai-recommendation-card">
                <h4>💡 {destination_storage} Recommendations</h4>
                <ul>
                    {"".join([f"<li>{rec}</li>" for rec in recommendations_for_storage])}
                </ul>
                <p><strong>Best Use Cases:</strong> {'High-performance workloads' if destination_storage == 'FSx_Lustre' else 'Windows applications' if destination_storage == 'FSx_Windows' else 'General purpose, cost-effective'}</p>
            </div>
            """, unsafe_allow_html=True)

def render_agent_scaling_tab(analysis: Dict, config: Dict):
    """Render dedicated agent scaling analysis tab with FSx destination awareness"""
    st.subheader("🤖 Agent Scaling Analysis & Optimization with Storage Destinations")
    
    agent_analysis = analysis.get('agent_analysis', {})
    optimal_recommendations = agent_analysis.get('optimal_recommendations', {})
    
    # Agent Configuration Overview with Destination Storage
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🔧 Current Agents",
            f"{config.get('number_of_agents', 1)} agents",
            delta=f"{agent_analysis.get('primary_tool', 'Unknown').upper()} {agent_analysis.get('agent_size', 'Unknown')}"
        )
    
    with col2:
        st.metric(
            "🗄️ Destination Storage",
            config.get('destination_storage_type', 'S3'),
            delta=f"Multiplier: {agent_analysis.get('storage_performance_multiplier', 1.0):.1f}x"
        )
    
    with col3:
        st.metric(
            "⚡ Total Throughput",
            f"{agent_analysis.get('total_max_throughput_mbps', 0):,.0f} Mbps",
            delta=f"Effective: {agent_analysis.get('total_effective_throughput', 0):,.0f} Mbps"
        )
    
    with col4:
        st.metric(
            "🎯 Scaling Efficiency",
            f"{agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%",
            delta=f"Overhead: {(agent_analysis.get('management_overhead', 1.0)-1)*100:.1f}%"
        )
    
    with col5:
        st.metric(
            "💰 Monthly Cost",
            f"${agent_analysis.get('monthly_cost', 0):,.0f}",
            delta=f"${agent_analysis.get('cost_per_hour', 0):.2f}/hour"
        )
    
    # Agent Configuration Details with Storage Impact
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Current Agent Configuration:**")
        
        agent_config = agent_analysis.get('agent_configuration', {})
        destination_storage = agent_analysis.get('destination_storage', 'S3')
        st.markdown(f"""
        <div class="agent-scaling-card">
            <h4>Current Setup Details</h4>
            <p><strong>Agent Type:</strong> {agent_analysis.get('primary_tool', 'Unknown').upper()}</p>
            <p><strong>Agent Size:</strong> {agent_analysis.get('agent_size', 'Unknown').title()}</p>
            <p><strong>Number of Agents:</strong> {agent_analysis.get('number_of_agents', 1)}</p>
            <p><strong>Destination Storage:</strong> {destination_storage}</p>
            <p><strong>Per-Agent vCPU:</strong> {agent_config.get('per_agent_spec', {}).get('vcpu', 'N/A')}</p>
            <p><strong>Per-Agent Memory:</strong> {agent_config.get('per_agent_spec', {}).get('memory_gb', 'N/A')} GB</p>
            <p><strong>Per-Agent Throughput:</strong> {agent_config.get('max_throughput_mbps_per_agent', 0):,.0f} Mbps</p>
            <p><strong>Storage Performance Bonus:</strong> {agent_config.get('storage_performance_multiplier', 1.0):.1f}x</p>
            <p><strong>Total Concurrent Tasks:</strong> {agent_config.get('total_concurrent_tasks', 0)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🎯 Optimal Configuration Recommendation:**")
        
        if optimal_recommendations.get('optimal_configuration'):
            optimal_config = optimal_recommendations['optimal_configuration']
            st.markdown(f"""
            <div class="agent-scaling-card">
                <h4>AI-Recommended Optimal Setup</h4>
                <p><strong>Optimal Agents:</strong> {optimal_config['configuration']['number_of_agents']}</p>
                <p><strong>Optimal Size:</strong> {optimal_config['configuration']['agent_size'].title()}</p>
                <p><strong>Recommended Destination:</strong> {optimal_config['configuration']['destination_storage']}</p>
                <p><strong>Total Throughput:</strong> {optimal_config['total_throughput_mbps']:,.0f} Mbps</p>
                <p><strong>Storage Performance:</strong> {optimal_config.get('storage_performance_multiplier', 1.0):.1f}x</p>
                <p><strong>Monthly Cost:</strong> ${optimal_config['total_cost_per_hour'] * 24 * 30:,.0f}</p>
                <p><strong>Overall Score:</strong> {optimal_config['overall_score']:.1f}/100</p>
                <p><strong>Efficiency Gain:</strong> {optimal_config['overall_score'] - agent_config.get('optimal_configuration', {}).get('efficiency_score', 0):.1f} points</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="agent-scaling-card">
                <h4>Configuration Analysis</h4>
                <p>Current configuration appears optimal for {destination_storage}</p>
                <p><strong>Efficiency Score:</strong> {agent_config.get('optimal_configuration', {}).get('efficiency_score', 0):.1f}/100</p>
                <p><strong>Management Complexity:</strong> {agent_config.get('optimal_configuration', {}).get('management_complexity', 'Unknown')}</p>
                <p><strong>Storage Optimization:</strong> {agent_config.get('optimal_configuration', {}).get('storage_optimization', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Throughput Analysis Chart with Storage Impact
    st.markdown("**📊 Throughput Analysis with Storage Impact:**")
    
    throughput_data = {
        'Component': ['Per-Agent Base', 'Per-Agent with Storage', 'Total Agent Capacity', 'Network Limit', 'Effective Throughput'],
        'Throughput (Mbps)': [
            agent_config.get('max_throughput_mbps_per_agent', 0),
            agent_config.get('max_throughput_mbps_per_agent', 0) * agent_config.get('storage_performance_multiplier', 1.0),
            agent_analysis.get('total_max_throughput_mbps', 0),
            analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0),
            agent_analysis.get('total_effective_throughput', 0)
        ],
        'Type': ['Base', 'Enhanced', 'Aggregate', 'Network', 'Effective']
    }
    
    fig_throughput = px.bar(
        throughput_data, 
        x='Component', 
        y='Throughput (Mbps)',
        color='Type',
        title=f"Agent vs Network Throughput Analysis ({destination_storage} Destination)",
        color_discrete_map={'Base': '#95a5a6', 'Enhanced': '#3498db', 'Aggregate': '#2ecc71', 'Network': '#e74c3c', 'Effective': '#9b59b6'}
    )
    
    st.plotly_chart(fig_throughput, use_container_width=True)
    
    # Scaling Efficiency Analysis with Storage Considerations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Scaling Efficiency Analysis:**")
        
        scaling_recommendations = agent_config.get('scaling_recommendations', [])
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>Scaling Assessment</h4>
            <p><strong>Current Efficiency:</strong> {agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%</p>
            <p><strong>Management Overhead:</strong> {(agent_analysis.get('management_overhead', 1.0)-1)*100:.1f}%</p>
            <p><strong>Storage Overhead:</strong> {(agent_analysis.get('storage_management_overhead', 1.0)-1)*100:.1f}%</p>
            <p><strong>Coordination Complexity:</strong> {agent_config.get('optimal_configuration', {}).get('management_complexity', 'Unknown')}</p>
            <ul>
                {"".join([f"<li>{rec}</li>" for rec in scaling_recommendations[:3]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🔍 Bottleneck Analysis:**")
        
        bottleneck = agent_analysis.get('bottleneck', 'unknown')
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>Performance Constraints</h4>
            <p><strong>Primary Bottleneck:</strong> {bottleneck.title()}</p>
            <p><strong>Severity:</strong> {bottleneck_severity.title()}</p>
            <p><strong>Throughput Impact:</strong> {agent_analysis.get('throughput_impact', 0)*100:.1f}%</p>
            <p><strong>Storage Impact:</strong> {destination_storage} provides {agent_analysis.get('storage_performance_multiplier', 1.0):.1f}x multiplier</p>
            <p><strong>Recommendation:</strong> {"Scale agents" if bottleneck == 'agents' else "Optimize network" if bottleneck == 'network' else "Monitor performance"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**💰 Cost Efficiency Analysis:**")
        
        cost_per_agent = agent_analysis.get('monthly_cost', 0) / config.get('number_of_agents', 1) if config.get('number_of_agents', 1) > 0 else 0
        cost_per_mbps = agent_analysis.get('monthly_cost', 0) / agent_analysis.get('total_effective_throughput', 1) if agent_analysis.get('total_effective_throughput', 0) > 0 else 0
        
        storage_cost_impact = {
            'S3': 1.0,
            'FSx_Windows': 1.1,
            'FSx_Lustre': 1.2
        }.get(destination_storage, 1.0)
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>Cost Efficiency Metrics</h4>
            <p><strong>Cost per Agent:</strong> ${cost_per_agent:,.0f}/month</p>
            <p><strong>Cost per Mbps:</strong> ${cost_per_mbps:.2f}/month</p>
            <p><strong>Total Monthly:</strong> ${agent_analysis.get('monthly_cost', 0):,.0f}</p>
            <p><strong>Storage Impact:</strong> {storage_cost_impact:.1f}x base cost</p>
            <p><strong>Efficiency Rating:</strong> {agent_config.get('optimal_configuration', {}).get('cost_efficiency', 'Unknown')}</p>
            <p><strong>Storage Optimization:</strong> {agent_config.get('optimal_configuration', {}).get('storage_optimization', 'Standard')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Optimization Recommendations with Storage Considerations
    ai_optimization = agent_analysis.get('ai_optimization', {})
    optimization_recommendations = ai_optimization.get('recommendations', [])
    
    if optimization_recommendations:
        st.markdown("**🤖 AI Agent Optimization Recommendations:**")
        
        for i, recommendation in enumerate(optimization_recommendations, 1):
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            timeframe = "Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"
            
            # Determine if recommendation is storage-specific
            storage_specific = any(storage in recommendation.lower() for storage in ['s3', 'fsx', 'lustre', 'windows'])
            
            st.markdown(f"""
            <div class="ai-recommendation-card">
                <h5>Optimization {i}: {recommendation}</h5>
                <p><strong>Implementation Complexity:</strong> {complexity}</p>
                <p><strong>Timeframe:</strong> {timeframe}</p>
                <p><strong>Expected Benefit:</strong> {ai_optimization.get('optimization_potential_percent', 0) // i}% improvement</p>
                <p><strong>Storage Relevance:</strong> {"High" if storage_specific else "General"}</p>
                <p><strong>Destination:</strong> {"Specific to " + destination_storage if storage_specific else "All destinations"}</p>
            </div>
            """)

async def main():
    """Enhanced main function with professional UI, detailed agent scaling analysis, and FSx destination support"""
    render_enhanced_header()
    
    # Get configuration
    config = render_enhanced_sidebar_controls()
    
    # Initialize enhanced analyzer
    analyzer = EnhancedMigrationAnalyzer()
    
    # Run analysis
    analysis_placeholder = st.empty()
    
    with analysis_placeholder.container():
        if config['enable_ai_analysis']:
            with st.spinner("🧠 Running comprehensive AI-powered migration analysis with agent scaling optimization and FSx destination analysis..."):
                try:
                    analysis = await analyzer.comprehensive_ai_migration_analysis(config)
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    # Fallback to simplified analysis
                    analysis = {
                        'api_status': APIStatus(anthropic_connected=False, aws_pricing_connected=False),
                        'onprem_performance': {'performance_score': 75, 'os_impact': {'total_efficiency': 0.85}},
                        'network_performance': {'network_quality_score': 80, 'effective_bandwidth_mbps': 1000, 'segments': [], 'destination_storage': config.get('destination_storage_type', 'S3')},
                        'migration_type': 'homogeneous' if config['source_database_engine'] == config['database_engine'] else 'heterogeneous',
                        'primary_tool': 'datasync' if config['source_database_engine'] == config['database_engine'] else 'dms',
                        'agent_analysis': {
                            'primary_tool': 'datasync' if config['source_database_engine'] == config['database_engine'] else 'dms',
                            'number_of_agents': config.get('number_of_agents', 1),
                            'destination_storage': config.get('destination_storage_type', 'S3'),
                            'total_effective_throughput': 500 * config.get('number_of_agents', 1),
                            'scaling_efficiency': 0.95 if config.get('number_of_agents', 1) <= 3 else 0.85,
                            'storage_performance_multiplier': {'S3': 1.0, 'FSx_Windows': 1.15, 'FSx_Lustre': 1.4}.get(config.get('destination_storage_type', 'S3'), 1.0),
                            'monthly_cost': 150 * config.get('number_of_agents', 1)
                        },
                        'migration_throughput_mbps': 500,
                        'estimated_migration_time_hours': 8,
                        'aws_sizing_recommendations': {'deployment_recommendation': {'recommendation': 'rds'}},
                        'cost_analysis': {'total_monthly_cost': 1500, 'destination_storage_type': config.get('destination_storage_type', 'S3'), 'destination_storage_cost': 200},
                        'fsx_comparisons': {},
                        'ai_overall_assessment': {'migration_readiness_score': 75, 'risk_level': 'Medium'}
                    }
        else:
            with st.spinner("🔬 Running standard migration analysis with agent scaling and FSx destination analysis..."):
                # Simplified analysis without AI
                analysis = {
                    'api_status': APIStatus(anthropic_connected=False, aws_pricing_connected=False),
                    'onprem_performance': {'performance_score': 75, 'os_impact': {'total_efficiency': 0.85}},
                    'network_performance': {'network_quality_score': 80, 'effective_bandwidth_mbps': 1000, 'segments': [], 'destination_storage': config.get('destination_storage_type', 'S3')},
                    'migration_type': 'homogeneous' if config['source_database_engine'] == config['database_engine'] else 'heterogeneous',
                    'primary_tool': 'datasync' if config['source_database_engine'] == config['database_engine'] else 'dms',
                    'agent_analysis': {
                        'primary_tool': 'datasync' if config['source_database_engine'] == config['database_engine'] else 'dms',
                        'number_of_agents': config.get('number_of_agents', 1),
                        'destination_storage': config.get('destination_storage_type', 'S3'),
                        'total_effective_throughput': 500 * config.get('number_of_agents', 1),
                        'scaling_efficiency': 0.95 if config.get('number_of_agents', 1) <= 3 else 0.85,
                        'storage_performance_multiplier': {'S3': 1.0, 'FSx_Windows': 1.15, 'FSx_Lustre': 1.4}.get(config.get('destination_storage_type', 'S3'), 1.0),
                        'monthly_cost': 150 * config.get('number_of_agents', 1)
                    },
                    'migration_throughput_mbps': 500,
                    'estimated_migration_time_hours': 8,
                    'aws_sizing_recommendations': {'deployment_recommendation': {'recommendation': 'rds'}},
                    'cost_analysis': {'total_monthly_cost': 1500, 'destination_storage_type': config.get('destination_storage_type', 'S3'), 'destination_storage_cost': 200},
                    'fsx_comparisons': {},
                    'ai_overall_assessment': {'migration_readiness_score': 75, 'risk_level': 'Medium'}
                }
    
    analysis_placeholder.empty()
    
    # Enhanced tabs with FSx destination comparison
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🧠 AI Insights & Analysis", 
        "🤖 Agent Scaling Analysis",
        "🗄️ FSx Destination Comparison",
        "🌐 Network Intelligence",
        "💰 Cost & Pricing Analysis",
        "💻 OS Performance Analysis",
        "📊 Migration Dashboard",
        "🎯 AWS Sizing & Configuration",
        "📄 Executive PDF Reports"
    ])
    
    with tab1:
        if config['enable_ai_analysis']:
            render_ai_insights_tab_enhanced(analysis, config)
        else:
            st.info("🤖 Enable AI Analysis in the sidebar for comprehensive migration insights")
    
    with tab2:
        render_agent_scaling_tab(analysis, config)
    
    with tab3:
        render_fsx_destination_comparison_tab(analysis, config)
    
    with tab4:
        render_network_intelligence_tab(analysis, config)
    
    with tab5:
        render_cost_pricing_tab(analysis, config)
    
    with tab6:
        render_os_performance_tab(analysis, config)
    
    with tab7:
        render_migration_dashboard_tab(analysis, config)
    
    with tab8:
        render_aws_sizing_tab(analysis, config)
    
    # Professional footer with FSx capabilities
    st.markdown("""
    <div class="enterprise-footer">
        <h4>🚀 AWS Enterprise Database Migration Analyzer AI v3.0</h4>
        <p>Powered by Anthropic Claude AI • Real-time AWS Integration • Professional Migration Analysis • Advanced Agent Scaling • FSx Destination Analysis</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            🔬 Advanced Network Intelligence • 🎯 AI-Driven Recommendations • 📊 Executive Reporting • 🤖 Multi-Agent Optimization • 🗄️ S3/FSx Comparisons
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_pdf_reports_tab(analysis: Dict, config: Dict):
    """Render PDF reports generation tab with FSx destination support"""
    st.subheader("📄 Executive PDF Reports")
    
    # PDF Generation Section
    st.markdown(f"""
    <div class="pdf-download-section">
        <h3>🎯 Generate Executive Migration Report</h3>
        <p>Create a comprehensive PDF report for stakeholders with all analysis results, recommendations, technical details, agent scaling analysis, and FSx destination comparisons.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📊 Generate Executive PDF Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating comprehensive PDF report with agent scaling analysis and FSx destination comparisons..."):
                try:
                    # Generate PDF report
                    pdf_generator = PDFReportGenerator()
                    pdf_bytes = pdf_generator.generate_executive_report(analysis, config)
                    
                    # Create download button
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    destination = config.get('destination_storage_type', 'S3')
                    filename = f"AWS_Migration_Analysis_Agent_Scaling_FSx_{destination}_{current_time}.pdf"
                    
                    st.success("✅ PDF Report Generated Successfully!")
                    
                    st.download_button(
                        label="📥 Download Executive Report PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    st.info("💡 PDF generation requires matplotlib. The analysis is still available in other tabs.")
    
    # Report Contents Preview
    st.markdown("**📋 Report Contents:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>📈 Executive Summary</h4>
            <ul>
                <li>Migration overview and key metrics</li>
                <li>AI readiness assessment</li>
                <li>Cost summary and ROI analysis</li>
                <li>Performance baseline evaluation</li>
                <li>Agent scaling configuration</li>
                <li>FSx destination storage analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>⚙️ Technical Analysis</h4>
            <ul>
                <li>Current performance breakdown</li>
                <li>Network path analysis</li>
                <li>OS performance impact</li>
                <li>Identified bottlenecks and solutions</li>
                <li>Agent throughput analysis</li>
                <li>Storage destination performance impact</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>🤖 Agent Scaling Analysis</h4>
            <ul>
                <li>Current agent configuration assessment</li>
                <li>Scaling efficiency analysis</li>
                <li>Optimal configuration recommendations</li>
                <li>Cost efficiency metrics</li>
                <li>Bottleneck identification</li>
                <li>Storage destination impact on agents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>🗄️ FSx Destination Comparison</h4>
            <ul>
                <li>S3 vs FSx for Windows vs FSx for Lustre</li>
                <li>Performance comparison analysis</li>
                <li>Cost impact assessment</li>
                <li>Migration time variations</li>
                <li>Complexity and risk factors</li>
                <li>Destination-specific recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>☁️ AWS Recommendations</h4>
            <ul>
                <li>Deployment recommendations (RDS vs EC2)</li>
                <li>Instance sizing and configuration</li>
                <li>Reader/writer setup</li>
                <li>AI complexity analysis</li>
                <li>Agent impact on AWS sizing</li>
                <li>Storage destination optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>📊 Cost & Risk Analysis</h4>
            <ul>
                <li>Detailed cost breakdown with agent costs</li>
                <li>FSx destination cost comparisons</li>
                <li>Financial projections and ROI</li>
                <li>Risk assessment matrix</li>
                <li>Recommended timeline and next steps</li>
                <li>Agent scaling cost optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Report Options
    st.markdown("**🔧 Report Customization:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        include_technical_details = st.checkbox("Include Technical Deep-Dive", value=True, help="Include detailed technical analysis and performance metrics")
    
    with col2:
        include_ai_insights = st.checkbox("Include AI Insights", value=True, help="Include all AI-generated recommendations and analysis")
    
    with col3:
        include_agent_analysis = st.checkbox("Include Agent Scaling Analysis", value=True, help="Include detailed agent configuration and optimization analysis")
    
    with col4:
        include_fsx_comparison = st.checkbox("Include FSx Destination Analysis", value=True, help="Include comprehensive FSx destination comparison")
    
    # Report Metrics
    st.markdown("**📊 Report Metrics:**")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        pages = 6 + (1 if include_fsx_comparison else 0)
        st.metric("📄 Pages", str(pages), delta="Executive format with FSx analysis")
    
    with metrics_col2:
        chart_count = 12 + (3 if include_technical_details else 0) + (2 if include_ai_insights else 0) + (3 if include_agent_analysis else 0) + (4 if include_fsx_comparison else 0)
        st.metric("📊 Charts & Graphs", str(chart_count), delta="Visual analysis")
    
    with metrics_col3:
        recommendation_count = len(analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('performance_recommendations', []))
        fsx_count = len(analysis.get('fsx_comparisons', {}))
        st.metric("💡 AI Recommendations", str(recommendation_count + fsx_count), delta="Including FSx insights")
    
    with metrics_col4:
        complexity_score = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)
        st.metric("🎯 AI Complexity", f"{complexity_score}/10", delta="Migration difficulty")

def render_network_intelligence_tab(analysis: Dict, config: Dict):
    """Render network intelligence analysis tab with AI insights"""
    st.subheader("🌐 Network Intelligence & Path Optimization")
    
    network_perf = analysis.get('network_performance', {})
    
    # Network Overview Dashboard
    st.markdown("**📊 Network Performance Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎯 Network Quality",
            f"{network_perf.get('network_quality_score', 0):.1f}/100",
            delta=f"AI Enhanced: {network_perf.get('ai_enhanced_quality_score', 0):.1f}"
        )
    
    with col2:
        st.metric(
            "⚡ Effective Bandwidth",
            f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps",
            delta=f"Utilization: {min(100, network_perf.get('effective_bandwidth_mbps', 0) / 10000 * 100):.1f}%"
        )
    
    with col3:
        st.metric(
            "🕐 Total Latency",
            f"{network_perf.get('total_latency_ms', 0):.1f} ms",
            delta=f"Reliability: {network_perf.get('total_reliability', 0)*100:.2f}%"
        )
    
    with col4:
        st.metric(
            "🗄️ Destination Storage",
            network_perf.get('destination_storage', 'S3'),
            delta=f"Bonus: +{network_perf.get('storage_performance_bonus', 0)}%"
        )
    
    with col5:
        ai_optimization = network_perf.get('ai_optimization_potential', 0)
        st.metric(
            "🤖 AI Optimization",
            f"{ai_optimization:.1f}%",
            delta="Improvement potential"
        )
    
    # Network Path Visualization
    st.markdown("**🗺️ Network Path Visualization:**")
    
    if network_perf.get('segments'):
        # Create network path diagram
        try:
            network_diagram = create_network_path_diagram(network_perf)
            st.plotly_chart(network_diagram, use_container_width=True)
        except Exception as e:
            st.warning(f"Network diagram could not be rendered: {str(e)}")
            
            # Fallback: Show path as table
            segments_data = []
            for i, segment in enumerate(network_perf.get('segments', []), 1):
                segments_data.append({
                    'Hop': i,
                    'Segment': segment['name'],
                    'Type': segment['connection_type'].replace('_', ' ').title(),
                    'Bandwidth (Mbps)': f"{segment.get('effective_bandwidth_mbps', 0):,.0f}",
                    'Latency (ms)': f"{segment.get('effective_latency_ms', 0):.1f}",
                    'Reliability': f"{segment['reliability']*100:.3f}%",
                    'Cost Factor': f"{segment['cost_factor']:.1f}x"
                })
            
            df_segments = pd.DataFrame(segments_data)
            st.dataframe(df_segments, use_container_width=True)
    else:
        st.info("Network path data not available in current analysis")
    
    # Detailed Network Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Network Performance Analysis:**")
        
        st.markdown(f"""
        <div class="network-intelligence-card">
            <h4>Performance Metrics</h4>
            <p><strong>Path Name:</strong> {network_perf.get('path_name', 'Unknown')}</p>
            <p><strong>Environment:</strong> {network_perf.get('environment', 'Unknown').title()}</p>
            <p><strong>OS Type:</strong> {network_perf.get('os_type', 'Unknown').title()}</p>
            <p><strong>Storage Type:</strong> {network_perf.get('storage_type', 'Unknown').title()}</p>
            <p><strong>Destination:</strong> {network_perf.get('destination_storage', 'S3')}</p>
            <p><strong>Network Quality Score:</strong> {network_perf.get('network_quality_score', 0):.1f}/100</p>
            <p><strong>AI Enhanced Score:</strong> {network_perf.get('ai_enhanced_quality_score', 0):.1f}/100</p>
            <p><strong>Cost Factor:</strong> {network_perf.get('total_cost_factor', 0):.1f}x</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🤖 AI Network Insights:**")
        
        ai_insights = network_perf.get('ai_insights', {})
        
        st.markdown(f"""
        <div class="network-intelligence-card">
            <h4>AI Analysis & Recommendations</h4>
            <p><strong>Performance Bottlenecks:</strong></p>
            <ul>
                {"".join([f"<li>{bottleneck}</li>" for bottleneck in ai_insights.get('performance_bottlenecks', ['No bottlenecks identified'])[:3]])}
            </ul>
            <p><strong>Optimization Opportunities:</strong></p>
            <ul>
                {"".join([f"<li>{opportunity}</li>" for opportunity in ai_insights.get('optimization_opportunities', ['Standard optimization'])[:3]])}
            </ul>
            <p><strong>Risk Factors:</strong></p>
            <ul>
                {"".join([f"<li>{risk}</li>" for risk in ai_insights.get('risk_factors', ['No significant risks'])[:2]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Network Optimization Recommendations
    st.markdown("**💡 Network Optimization Recommendations:**")
    
    recommendations = ai_insights.get('recommended_improvements', [])
    if recommendations:
        for i, recommendation in enumerate(recommendations, 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            
            st.markdown(f"""
            <div class="detailed-analysis-section">
                <h5>Recommendation {i}: {recommendation}</h5>
                <p><strong>Expected Impact:</strong> {impact}</p>
                <p><strong>Implementation Complexity:</strong> {complexity}</p>
                <p><strong>Priority:</strong> {"Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"}</p>
            </div>
            """)
    else:
        st.info("Network appears optimally configured for current requirements")

def render_cost_pricing_tab(analysis: Dict, config: Dict):
    """Render comprehensive cost and pricing analysis tab"""
    st.subheader("💰 Live AWS Pricing & Cost Analysis")
    
    cost_analysis = analysis.get('cost_analysis', {})
    
    # Cost Overview Dashboard
    st.markdown("**💸 Cost Overview Dashboard:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_monthly = cost_analysis.get('total_monthly_cost', 0)
        st.metric(
            "💰 Total Monthly Cost",
            f"${total_monthly:,.0f}",
            delta=f"Annual: ${total_monthly * 12:,.0f}"
        )
    
    with col2:
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        st.metric(
            "🔄 Migration Cost",
            f"${one_time_cost:,.0f}",
            delta="One-time"
        )
    
    with col3:
        savings = cost_analysis.get('estimated_monthly_savings', 0)
        st.metric(
            "📈 Monthly Savings",
            f"${savings:,.0f}",
            delta=f"Annual: ${savings * 12:,.0f}"
        )
    
    with col4:
        roi_months = cost_analysis.get('roi_months', 0)
        st.metric(
            "⏰ ROI Timeline",
            f"{roi_months} months" if roi_months else "TBD",
            delta=f"Break-even: Year {roi_months / 12:.1f}" if roi_months else "Calculate needed"
        )
    
    with col5:
        agent_cost = cost_analysis.get('agent_cost', 0)
        num_agents = config.get('number_of_agents', 1)
        st.metric(
            "🤖 Agent Costs",
            f"${agent_cost:,.0f}/mo",
            delta=f"${agent_cost / num_agents:.0f} per agent" if num_agents > 0 else ""
        )
    
    # Cost Breakdown Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Monthly Cost Breakdown:**")
        
        # Create cost breakdown pie chart
        cost_breakdown = {
            'Compute': cost_analysis.get('aws_compute_cost', 0),
            'Storage': cost_analysis.get('aws_storage_cost', 0),
            'Network': cost_analysis.get('network_cost', 0),
            'Agents': cost_analysis.get('agent_cost', 0),
            'Destination Storage': cost_analysis.get('destination_storage_cost', 0),
            'OS Licensing': cost_analysis.get('os_licensing_cost', 0),
            'Management': cost_analysis.get('management_cost', 0)
        }
        
        # Filter out zero values
        cost_breakdown = {k: v for k, v in cost_breakdown.items() if v > 0}
        
        if cost_breakdown:
            fig_pie = px.pie(
                values=list(cost_breakdown.values()),
                names=list(cost_breakdown.keys()),
                title="Monthly Cost Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Cost breakdown data not available")
    
    with col2:
        st.markdown("**📈 Cost Projections:**")
        
        # Create cost projections chart
        monthly_cost = cost_analysis.get('total_monthly_cost', 0)
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        
        projections = {
            '6 Months': monthly_cost * 6 + one_time_cost,
            '1 Year': monthly_cost * 12 + one_time_cost,
            '2 Years': monthly_cost * 24 + one_time_cost,
            '3 Years': monthly_cost * 36 + one_time_cost,
            '5 Years': monthly_cost * 60 + one_time_cost
        }
        
        fig_bar = px.bar(
            x=list(projections.keys()),
            y=list(projections.values()),
            title="Total Cost Projections",
            labels={'x': 'Timeline', 'y': 'Total Cost ($)'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed Cost Analysis
    st.markdown("**🔍 Detailed Cost Analysis:**")
    
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    
    with detail_col1:
        st.markdown(f"""
        <div class="live-pricing-card">
            <h4>💻 Compute Costs</h4>
            <p><strong>AWS Compute:</strong> ${cost_analysis.get('aws_compute_cost', 0):,.0f}/month</p>
            <p><strong>AWS Storage:</strong> ${cost_analysis.get('aws_storage_cost', 0):,.0f}/month</p>
            <p><strong>OS Licensing:</strong> ${cost_analysis.get('os_licensing_cost', 0):,.0f}/month</p>
            <p><strong>Management:</strong> ${cost_analysis.get('management_cost', 0):,.0f}/month</p>
            <p><strong>Subtotal:</strong> ${cost_analysis.get('aws_compute_cost', 0) + cost_analysis.get('aws_storage_cost', 0) + cost_analysis.get('os_licensing_cost', 0) + cost_analysis.get('management_cost', 0):,.0f}/month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with detail_col2:
        st.markdown(f"""
        <div class="live-pricing-card">
            <h4>🔄 Migration & Agent Costs</h4>
            <p><strong>Agent Base Cost:</strong> ${cost_analysis.get('agent_base_cost', 0):,.0f}/month</p>
            <p><strong>Agent Setup:</strong> ${cost_analysis.get('agent_setup_cost', 0):,.0f} (one-time)</p>
            <p><strong>Coordination Cost:</strong> ${cost_analysis.get('agent_coordination_cost', 0):,.0f} (one-time)</p>
            <p><strong>Storage Setup:</strong> ${cost_analysis.get('storage_setup_cost', 0):,.0f} (one-time)</p>
            <p><strong>Total Migration:</strong> ${cost_analysis.get('one_time_migration_cost', 0):,.0f} (one-time)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with detail_col3:
        st.markdown(f"""
        <div class="live-pricing-card">
            <h4>🗄️ Storage & Network Costs</h4>
            <p><strong>Destination Storage:</strong> ${cost_analysis.get('destination_storage_cost', 0):,.0f}/month</p>
            <p><strong>Storage Type:</strong> {cost_analysis.get('destination_storage_type', 'S3')}</p>
            <p><strong>Network Costs:</strong> ${cost_analysis.get('network_cost', 0):,.0f}/month</p>
            <p><strong>Data Transfer:</strong> Included in network</p>
            <p><strong>Subtotal:</strong> ${cost_analysis.get('destination_storage_cost', 0) + cost_analysis.get('network_cost', 0):,.0f}/month</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cost Optimization Analysis
    st.markdown("**🎯 Cost Optimization Analysis:**")
    
    ai_cost_insights = cost_analysis.get('ai_cost_insights', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>💡 AI Cost Optimization Insights</h4>
            <p><strong>Optimization Factor:</strong> {ai_cost_insights.get('ai_optimization_factor', 0)*100:.1f}%</p>
            <p><strong>Complexity Multiplier:</strong> {ai_cost_insights.get('complexity_multiplier', 1.0):.2f}x</p>
            <p><strong>Management Reduction:</strong> {ai_cost_insights.get('management_reduction', 0)*100:.1f}%</p>
            <p><strong>Agent Efficiency Bonus:</strong> {ai_cost_insights.get('agent_efficiency_bonus', 0)*100:.1f}%</p>
            <p><strong>Storage Efficiency Bonus:</strong> {ai_cost_insights.get('storage_efficiency_bonus', 0)*100:.1f}%</p>
            <p><strong>Additional Savings Potential:</strong> {ai_cost_insights.get('potential_additional_savings', '0%')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>📊 ROI & Financial Analysis</h4>
            <p><strong>Current Monthly Cost (Est.):</strong> ${total_monthly * 1.2:,.0f}</p>
            <p><strong>AWS Monthly Cost:</strong> ${total_monthly:,.0f}</p>
            <p><strong>Monthly Savings:</strong> ${cost_analysis.get('estimated_monthly_savings', 0):,.0f}</p>
            <p><strong>Annual Savings:</strong> ${cost_analysis.get('estimated_monthly_savings', 0) * 12:,.0f}</p>
            <p><strong>Payback Period:</strong> {roi_months} months</p>
            <p><strong>3-Year ROI:</strong> {((cost_analysis.get('estimated_monthly_savings', 0) * 36 - one_time_cost) / one_time_cost * 100):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time Pricing Data
    st.markdown("**💲 Real-time AWS Pricing Data:**")
    
    pricing_data = analysis.get('aws_sizing_recommendations', {}).get('pricing_data', {})
    
    if pricing_data.get('data_source') == 'aws_api':
        st.success("✅ Using real-time AWS pricing data")
        st.caption(f"Last updated: {pricing_data.get('last_updated', 'Unknown')}")
    else:
        st.warning("⚠️ Using fallback pricing data - AWS API not available")
    
    # Create pricing comparison table
    if pricing_data:
        tab1, tab2, tab3 = st.tabs(["🖥️ EC2 Instances", "🗄️ RDS Instances", "💾 Storage Types"])
        
        with tab1:
            ec2_instances = pricing_data.get('ec2_instances', {})
            if ec2_instances:
                ec2_data = []
                for instance, specs in ec2_instances.items():
                    ec2_data.append({
                        'Instance Type': instance,
                        'vCPU': specs.get('vcpu', 'N/A'),
                        'Memory (GB)': specs.get('memory', 'N/A'),
                        'Cost per Hour': f"${specs.get('cost_per_hour', 0):.4f}",
                        'Monthly Cost': f"${specs.get('cost_per_hour', 0) * 24 * 30:.0f}"
                    })
                
                df_ec2 = pd.DataFrame(ec2_data)
                st.dataframe(df_ec2, use_container_width=True)
        
        with tab2:
            rds_instances = pricing_data.get('rds_instances', {})
            if rds_instances:
                rds_data = []
                for instance, specs in rds_instances.items():
                    rds_data.append({
                        'Instance Type': instance,
                        'vCPU': specs.get('vcpu', 'N/A'),
                        'Memory (GB)': specs.get('memory', 'N/A'),
                        'Cost per Hour': f"${specs.get('cost_per_hour', 0):.4f}",
                        'Monthly Cost': f"${specs.get('cost_per_hour', 0) * 24 * 30:.0f}"
                    })
                
                df_rds = pd.DataFrame(rds_data)
                st.dataframe(df_rds, use_container_width=True)
        
        with tab3:
            storage = pricing_data.get('storage', {})
            if storage:
                storage_data = []
                for storage_type, specs in storage.items():
                    storage_data.append({
                        'Storage Type': storage_type.upper(),
                        'Cost per GB/Month': f"${specs.get('cost_per_gb_month', 0):.3f}",
                        'IOPS Included': specs.get('iops_included', 'N/A'),
                        'Cost per IOPS/Month': f"${specs.get('cost_per_iops_month', 0):.3f}" if specs.get('cost_per_iops_month') else 'N/A'
                    })
                
                df_storage = pd.DataFrame(storage_data)
                st.dataframe(df_storage, use_container_width=True)

def render_os_performance_tab(analysis: Dict, config: Dict):
    """Render OS performance analysis tab with detailed metrics"""
    st.subheader("💻 Operating System Performance Analysis")
    
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    
    # OS Overview
    st.markdown("**🖥️ Operating System Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💻 Current OS",
            os_impact.get('name', 'Unknown'),
            delta=f"Platform: {config.get('server_type', 'Unknown').title()}"
        )
    
    with col2:
        st.metric(
            "⚡ Total Efficiency",
            f"{os_impact.get('total_efficiency', 0)*100:.1f}%",
            delta=f"Base: {os_impact.get('base_efficiency', 0)*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "🗄️ DB Optimization",
            f"{os_impact.get('db_optimization', 0)*100:.1f}%",
            delta=f"Engine: {config.get('database_engine', 'Unknown').upper()}"
        )
    
    with col4:
        st.metric(
            "💰 Licensing Factor",
            f"{os_impact.get('licensing_cost_factor', 1.0):.1f}x",
            delta=f"Complexity: {os_impact.get('management_complexity', 0)*100:.0f}%"
        )
    
    with col5:
        virt_overhead = os_impact.get('virtualization_overhead', 0)
        st.metric(
            "☁️ Virtualization",
            f"{virt_overhead*100:.1f}%" if config.get('server_type') == 'vmware' else "N/A",
            delta="Overhead" if config.get('server_type') == 'vmware' else "Physical"
        )
    
    # OS Performance Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 OS Performance Metrics:**")
        
        # Create radar chart for OS performance
        performance_metrics = {
            'CPU Efficiency': os_impact.get('cpu_efficiency', 0) * 100,
            'Memory Efficiency': os_impact.get('memory_efficiency', 0) * 100,
            'I/O Efficiency': os_impact.get('io_efficiency', 0) * 100,
            'Network Efficiency': os_impact.get('network_efficiency', 0) * 100,
            'DB Optimization': os_impact.get('db_optimization', 0) * 100
        }
        
        fig_radar = go.Figure()
        
        categories = list(performance_metrics.keys())
        values = list(performance_metrics.values())
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='OS Performance',
            line_color='#3498db'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="OS Performance Profile"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.markdown("**🤖 AI OS Insights:**")
        
        ai_insights = os_impact.get('ai_insights', {})
        
        st.markdown(f"""
        <div class="os-performance-enhanced-card">
            <h4>AI Analysis of {os_impact.get('name', 'Current OS')}</h4>
            <p><strong>Strengths:</strong></p>
            <ul>
                {"".join([f"<li>{strength}</li>" for strength in ai_insights.get('strengths', ['General purpose OS'])[:3]])}
            </ul>
            <p><strong>Weaknesses:</strong></p>
            <ul>
                {"".join([f"<li>{weakness}</li>" for weakness in ai_insights.get('weaknesses', ['No significant issues'])[:3]])}
            </ul>
            <p><strong>Migration Considerations:</strong></p>
            <ul>
                {"".join([f"<li>{consideration}</li>" for consideration in ai_insights.get('migration_considerations', ['Standard migration'])[:3]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Database Engine Optimization
    st.markdown("**🗄️ Database Engine Optimization Analysis:**")
    
    database_optimizations = os_impact.get('database_optimizations', {})
    
    if database_optimizations:
        opt_data = []
        current_engine = config.get('database_engine', 'unknown')
        
        for engine, optimization in database_optimizations.items():
            opt_data.append({
                'Database Engine': engine.upper(),
                'Optimization Score': f"{optimization*100:.1f}%",
                'Performance Rating': 'Excellent' if optimization > 0.95 else 'Very Good' if optimization > 0.90 else 'Good' if optimization > 0.85 else 'Fair',
                'Current Selection': '✅' if engine == current_engine else ''
            })
        
        df_opt = pd.DataFrame(opt_data)
        st.dataframe(df_opt, use_container_width=True)
    
    # OS Comparison Analysis
    st.markdown("**⚖️ OS Comparison Analysis:**")
    
    # Create comparison with other OS options
    os_manager = OSPerformanceManager()
    comparison_data = []
    
    current_os = config.get('operating_system')
    current_platform = config.get('server_type')
    current_db_engine = config.get('database_engine')
    
    for os_name, os_config in os_manager.operating_systems.items():
        os_perf = os_manager.calculate_os_performance_impact(os_name, current_platform, current_db_engine)
        
        comparison_data.append({
            'Operating System': os_config['name'],
            'Total Efficiency': f"{os_perf['total_efficiency']*100:.1f}%",
            'CPU Efficiency': f"{os_perf['cpu_efficiency']*100:.1f}%",
            'Memory Efficiency': f"{os_perf['memory_efficiency']*100:.1f}%",
            'I/O Efficiency': f"{os_perf['io_efficiency']*100:.1f}%",
            'Licensing Cost': f"{os_perf['licensing_cost_factor']:.1f}x",
            'Current Choice': '✅' if os_name == current_os else ''
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Performance Impact Analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🎯 Performance Impact Assessment</h4>
            <p><strong>Base Efficiency:</strong> {os_impact.get('base_efficiency', 0)*100:.1f}%</p>
            <p><strong>Database Optimization:</strong> {os_impact.get('db_optimization', 0)*100:.1f}%</p>
            <p><strong>Platform Optimization:</strong> {os_impact.get('platform_optimization', 1.0)*100:.1f}%</p>
            <p><strong>Overall Impact:</strong> {((os_impact.get('total_efficiency', 0) - 0.8) / 0.2 * 100):.1f}% above baseline</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>💰 Cost Impact Analysis</h4>
            <p><strong>Licensing Cost Factor:</strong> {os_impact.get('licensing_cost_factor', 1.0):.1f}x</p>
            <p><strong>Monthly Licensing Est.:</strong> ${os_impact.get('licensing_cost_factor', 1.0) * 150:.0f}</p>
            <p><strong>Management Complexity:</strong> {os_impact.get('management_complexity', 0)*100:.0f}%</p>
            <p><strong>Security Overhead:</strong> {os_impact.get('security_overhead', 0)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🔧 Migration Recommendations</h4>
            <p><strong>Current OS Suitability:</strong> {"Excellent" if os_impact.get('total_efficiency', 0) > 0.9 else "Good" if os_impact.get('total_efficiency', 0) > 0.8 else "Fair"}</p>
            <p><strong>Migration Complexity:</strong> {"Low" if 'windows' in current_os and 'windows' in current_os else "Medium"}</p>
            <p><strong>Recommended Action:</strong> {"Keep current OS" if os_impact.get('total_efficiency', 0) > 0.85 else "Consider optimization"}</p>
        </div>
        """, unsafe_allow_html=True)

def render_migration_dashboard_tab(analysis: Dict, config: Dict):
    """Render comprehensive migration dashboard with key metrics and visualizations"""
    st.subheader("📊 Enhanced Migration Performance Dashboard")
    
    # Executive Summary Dashboard
    st.markdown("**🎯 Executive Migration Summary:**")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        st.metric(
            "🎯 Readiness Score",
            f"{readiness_score:.0f}/100",
            delta=analysis.get('ai_overall_assessment', {}).get('risk_level', 'Unknown')
        )
    
    with col2:
        migration_time = analysis.get('estimated_migration_time_hours', 0)
        st.metric(
            "⏱️ Migration Time",
            f"{migration_time:.1f} hours",
            delta=f"Window: {config.get('downtime_tolerance_minutes', 60)} min"
        )
    
    with col3:
        throughput = analysis.get('migration_throughput_mbps', 0)
        st.metric(
            "🚀 Throughput",
            f"{throughput:,.0f} Mbps",
            delta=f"Agents: {config.get('number_of_agents', 1)}"
        )
    
    with col4:
        monthly_cost = analysis.get('cost_analysis', {}).get('total_monthly_cost', 0)
        st.metric(
            "💰 Monthly Cost",
            f"${monthly_cost:,.0f}",
            delta=f"ROI: {analysis.get('cost_analysis', {}).get('roi_months', 'TBD')} months"
        )
    
    with col5:
        destination = config.get('destination_storage_type', 'S3')
        agent_efficiency = analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0)
        st.metric(
            "🗄️ Destination",
            destination,
            delta=f"Efficiency: {agent_efficiency*100:.1f}%"
        )
    
    with col6:
        complexity = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)
        confidence = analysis.get('ai_overall_assessment', {}).get('ai_confidence', 0.5)
        st.metric(
            "🤖 AI Confidence",
            f"{confidence*100:.1f}%",
            delta=f"Complexity: {complexity:.1f}/10"
        )
    
    # Performance Overview Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Performance Analysis:**")
        
        # Current vs Target Performance
        onprem_perf = analysis.get('onprem_performance', {}).get('overall_performance', {})
        
        performance_comparison = {
            'Metric': ['CPU', 'Memory', 'Storage', 'Network', 'Database'],
            'Current Score': [
                onprem_perf.get('cpu_score', 0),
                onprem_perf.get('memory_score', 0),
                onprem_perf.get('storage_score', 0),
                onprem_perf.get('network_score', 0),
                onprem_perf.get('database_score', 0)
            ],
            'Target (AWS)': [85, 90, 95, 90, 88]  # Estimated AWS performance
        }
        
        fig_perf = px.bar(
            performance_comparison,
            x='Metric',
            y=['Current Score', 'Target (AWS)'],
            title="Current vs AWS Target Performance",
            barmode='group'
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        st.markdown("**🔄 Migration Timeline:**")
        
        # Timeline breakdown
        timeline = analysis.get('ai_overall_assessment', {}).get('timeline_recommendation', {})
        
        timeline_data = {
            'Phase': ['Planning', 'Testing', 'Migration', 'Validation'],
            'Duration (Weeks)': [
                timeline.get('planning_phase_weeks', 2),
                timeline.get('testing_phase_weeks', 3),
                timeline.get('migration_window_hours', 24) / (7 * 24),  # Convert to weeks
                1  # Validation week
            ]
        }
        
        fig_timeline = px.bar(
            timeline_data,
            x='Phase',
            y='Duration (Weeks)',
            title="Project Timeline Breakdown",
            color='Duration (Weeks)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Agent and Network Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Agent Performance Analysis:**")
        
        agent_analysis = analysis.get('agent_analysis', {})
        
        # Agent throughput breakdown
        agent_data = {
            'Component': ['Per Agent', 'Total Agents', 'Network Limit', 'Effective'],
            'Throughput (Mbps)': [
                agent_analysis.get('total_max_throughput_mbps', 0) / config.get('number_of_agents', 1),
                agent_analysis.get('total_max_throughput_mbps', 0),
                analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0),
                agent_analysis.get('total_effective_throughput', 0)
            ]
        }
        
        fig_agent = px.bar(
            agent_data,
            x='Component',
            y='Throughput (Mbps)',
            title=f"Agent Throughput Analysis ({config.get('number_of_agents', 1)} agents)",
            color='Throughput (Mbps)',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_agent, use_container_width=True)
    
    with col2:
        st.markdown("**🌐 Network Performance Analysis:**")
        
        network_perf = analysis.get('network_performance', {})
        
        # Network metrics
        network_data = {
            'Metric': ['Quality Score', 'AI Enhanced', 'Bandwidth Util', 'Reliability'],
            'Score/Percentage': [
                network_perf.get('network_quality_score', 0),
                network_perf.get('ai_enhanced_quality_score', 0),
                min(100, network_perf.get('effective_bandwidth_mbps', 0) / 100),
                network_perf.get('total_reliability', 0) * 100
            ]
        }
        
        fig_network = px.bar(
            network_data,
            x='Metric',
            y='Score/Percentage',
            title="Network Performance Metrics",
            color='Score/Percentage',
            color_continuous_scale='greens'
        )
        st.plotly_chart(fig_network, use_container_width=True)
    
    # Cost and ROI Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Cost Analysis Dashboard:**")
        
        cost_analysis = analysis.get('cost_analysis', {})
        
        # Cost breakdown
        cost_categories = {
            'Compute': cost_analysis.get('aws_compute_cost', 0),
            'Storage': cost_analysis.get('aws_storage_cost', 0) + cost_analysis.get('destination_storage_cost', 0),
            'Agents': cost_analysis.get('agent_cost', 0),
            'Network': cost_analysis.get('network_cost', 0),
            'Other': cost_analysis.get('os_licensing_cost', 0) + cost_analysis.get('management_cost', 0)
        }
        
        fig_cost = px.pie(
            values=list(cost_categories.values()),
            names=list(cost_categories.keys()),
            title="Monthly Cost Distribution"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        st.markdown("**📈 ROI Projection:**")
        
        # ROI over time
        monthly_savings = cost_analysis.get('estimated_monthly_savings', 0)
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        
        months = list(range(0, 37, 6))  # 0 to 36 months
        cumulative_savings = []
        
        for month in months:
            if month == 0:
                cumulative_savings.append(-one_time_cost)
            else:
                cumulative_savings.append((monthly_savings * month) - one_time_cost)
        
        roi_data = {
            'Months': months,
            'Cumulative Savings ($)': cumulative_savings
        }
        
        fig_roi = px.line(
            roi_data,
            x='Months',
            y='Cumulative Savings ($)',
            title="ROI Projection Over Time",
            markers=True
        )
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        st.plotly_chart(fig_roi, use_container_width=True)
    
    # Risk and Readiness Assessment
    st.markdown("**⚠️ Risk and Readiness Assessment:**")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        ai_assessment = analysis.get('ai_overall_assessment', {})
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🎯 Migration Readiness</h4>
            <p><strong>Readiness Score:</strong> {ai_assessment.get('migration_readiness_score', 0):.0f}/100</p>
            <p><strong>Success Probability:</strong> {ai_assessment.get('success_probability', 0):.0f}%</p>
            <p><strong>Risk Level:</strong> {ai_assessment.get('risk_level', 'Unknown')}</p>
            <p><strong>Recommended Approach:</strong> {ai_assessment.get('timeline_recommendation', {}).get('recommended_approach', 'Unknown').replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col2:
        agent_impact = ai_assessment.get('agent_scaling_impact', {})
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🤖 Agent Scaling Assessment</h4>
            <p><strong>Scaling Efficiency:</strong> {agent_impact.get('scaling_efficiency', 0):.1f}%</p>
            <p><strong>Optimal Agents:</strong> {agent_impact.get('optimal_agents', config.get('number_of_agents', 1))}</p>
            <p><strong>Current Agents:</strong> {agent_impact.get('current_agents', config.get('number_of_agents', 1))}</p>
            <p><strong>Efficiency Bonus:</strong> {agent_impact.get('efficiency_bonus', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col3:
        storage_impact = ai_assessment.get('destination_storage_impact', {})
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🗄️ Storage Destination Impact</h4>
            <p><strong>Storage Type:</strong> {storage_impact.get('storage_type', 'Unknown')}</p>
            <p><strong>Performance Bonus:</strong> +{storage_impact.get('performance_bonus', 0)}%</p>
            <p><strong>Performance Multiplier:</strong> {storage_impact.get('storage_performance_multiplier', 1.0):.1f}x</p>
            <p><strong>Optimization Level:</strong> {"High" if storage_impact.get('performance_multiplier', 1.0) > 1.2 else "Medium" if storage_impact.get('performance_multiplier', 1.0) > 1.0 else "Standard"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Recommendations Summary
    st.markdown("**💡 Key Recommendations Summary:**")
    
    recommendations = ai_assessment.get('recommended_next_steps', [])
    if recommendations:
        for i, rec in enumerate(recommendations[:4], 1):
            priority = "🔴 High" if i <= 2 else "🟡 Medium"
            st.markdown(f"**{priority} Priority:** {rec}")
    
    # Migration Health Score
    st.markdown("**🏥 Migration Health Score:**")
    
    health_factors = {
        'Performance Readiness': min(100, analysis.get('onprem_performance', {}).get('performance_score', 0)),
        'Network Quality': network_perf.get('ai_enhanced_quality_score', 0),
        'Agent Optimization': agent_analysis.get('scaling_efficiency', 1.0) * 100,
        'Cost Efficiency': min(100, (monthly_savings / monthly_cost * 100)) if monthly_cost > 0 else 0,
        'Risk Mitigation': readiness_score
    }
    
    fig_health = px.bar(
        x=list(health_factors.keys()),
        y=list(health_factors.values()),
        title="Migration Health Score Breakdown",
        color=list(health_factors.values()),
        color_continuous_scale='RdYlGn',
        labels={'x': 'Factor', 'y': 'Score'}
    )
    
    # Add benchmark line at 80%
    fig_health.add_hline(y=80, line_dash="dash", line_color="blue", annotation_text="Target: 80%")
    
    st.plotly_chart(fig_health, use_container_width=True)

def render_aws_sizing_tab(analysis: Dict, config: Dict):
    """Render AWS sizing and configuration recommendations tab"""
    st.subheader("🎯 AWS Sizing & Configuration Recommendations")
    
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    deployment_rec = aws_sizing.get('deployment_recommendation', {})
    
    # Deployment Recommendation Overview
    st.markdown("**☁️ Deployment Recommendation:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        recommendation = deployment_rec.get('recommendation', 'unknown').upper()
        confidence = deployment_rec.get('confidence', 0)
        st.metric(
            "🎯 Recommended Deployment",
            recommendation,
            delta=f"Confidence: {confidence*100:.1f}%"
        )
    
    with col2:
        rds_score = deployment_rec.get('rds_score', 0)
        ec2_score = deployment_rec.get('ec2_score', 0)
        st.metric(
            "📊 RDS Score",
            f"{rds_score:.0f}",
            delta=f"EC2: {ec2_score:.0f}"
        )
    
    with col3:
        ai_analysis = aws_sizing.get('ai_analysis', {})
        complexity = ai_analysis.get('ai_complexity_score', 6)
        st.metric(
            "🤖 AI Complexity",
            f"{complexity:.1f}/10",
            delta=ai_analysis.get('confidence_level', 'medium').title()
        )
    
    with col4:
        if recommendation == 'RDS':
            monthly_cost = aws_sizing.get('rds_recommendations', {}).get('total_monthly_cost', 0)
        else:
            monthly_cost = aws_sizing.get('ec2_recommendations', {}).get('total_monthly_cost', 0)
        st.metric(
            "💰 Monthly Cost",
            f"${monthly_cost:,.0f}",
            delta="Compute + Storage"
        )
    
    with col5:
        reader_writer = aws_sizing.get('reader_writer_config', {})
        total_instances = reader_writer.get('total_instances', 1)
        st.metric(
            "🖥️ Total Instances",
            f"{total_instances}",
            delta=f"Writers: {reader_writer.get('writers', 1)}, Readers: {reader_writer.get('readers', 0)}"
        )
    
    # Detailed Sizing Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        if recommendation == 'RDS':
            st.markdown("**🗄️ RDS Recommended Configuration:**")
            
            rds_rec = aws_sizing.get('rds_recommendations', {})
            
            st.markdown(f"""
            <div class="aws-sizing-card">
                <h4>Amazon RDS Managed Service</h4>
                <p><strong>Instance Type:</strong> {rds_rec.get('primary_instance', 'N/A')}</p>
                <p><strong>vCPU:</strong> {rds_rec.get('instance_specs', {}).get('vcpu', 'N/A')}</p>
                <p><strong>Memory:</strong> {rds_rec.get('instance_specs', {}).get('memory', 'N/A')} GB</p>
                <p><strong>Storage:</strong> {rds_rec.get('storage_size_gb', 0):,.0f} GB</p>
                <p><strong>Storage Type:</strong> {rds_rec.get('storage_type', 'gp3').upper()}</p>
                <p><strong>Multi-AZ:</strong> {'Yes' if rds_rec.get('multi_az', False) else 'No'}</p>
                <p><strong>Backup Retention:</strong> {rds_rec.get('backup_retention_days', 7)} days</p>
                <p><strong>Monthly Instance Cost:</strong> ${rds_rec.get('monthly_instance_cost', 0):,.0f}</p>
                <p><strong>Monthly Storage Cost:</strong> ${rds_rec.get('monthly_storage_cost', 0):,.0f}</p>
                <p><strong>Total Monthly Cost:</strong> ${rds_rec.get('total_monthly_cost', 0):,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("**🖥️ EC2 Recommended Configuration:**")
            
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            
            st.markdown(f"""
            <div class="aws-sizing-card">
                <h4>Amazon EC2 Self-Managed</h4>
                <p><strong>Instance Type:</strong> {ec2_rec.get('primary_instance', 'N/A')}</p>
                <p><strong>vCPU:</strong> {ec2_rec.get('instance_specs', {}).get('vcpu', 'N/A')}</p>
                <p><strong>Memory:</strong> {ec2_rec.get('instance_specs', {}).get('memory', 'N/A')} GB</p>
                <p><strong>Storage:</strong> {ec2_rec.get('storage_size_gb', 0):,.0f} GB</p>
                <p><strong>Storage Type:</strong> {ec2_rec.get('storage_type', 'gp3').upper()}</p>
                <p><strong>EBS Optimized:</strong> {'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'}</p>
                <p><strong>Enhanced Networking:</strong> {'Yes' if ec2_rec.get('enhanced_networking', False) else 'No'}</p>
                <p><strong>Monthly Instance Cost:</strong> ${ec2_rec.get('monthly_instance_cost', 0):,.0f}</p>
                <p><strong>Monthly Storage Cost:</strong> ${ec2_rec.get('monthly_storage_cost', 0):,.0f}</p>
                <p><strong>Total Monthly Cost:</strong> ${ec2_rec.get('total_monthly_cost', 0):,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**🎯 AI Sizing Factors:**")
        
        if recommendation == 'RDS':
            sizing_factors = aws_sizing.get('rds_recommendations', {}).get('ai_sizing_factors', {})
        else:
            sizing_factors = aws_sizing.get('ec2_recommendations', {}).get('ai_sizing_factors', {})
        
        st.markdown(f"""
        <div class="aws-sizing-card">
            <h4>AI-Enhanced Sizing Analysis</h4>
            <p><strong>Complexity Multiplier:</strong> {sizing_factors.get('complexity_multiplier', 1.0):.2f}x</p>
            <p><strong>Agent Scaling Factor:</strong> {sizing_factors.get('agent_scaling_factor', 1.0):.2f}x</p>
            <p><strong>AI Complexity Score:</strong> {sizing_factors.get('ai_complexity_score', 6):.1f}/10</p>
            <p><strong>Storage Multiplier:</strong> {sizing_factors.get('storage_multiplier', 1.5):.1f}x</p>
            <p><strong>Database Size:</strong> {config.get('database_size_gb', 0):,} GB</p>
            <p><strong>Performance Requirement:</strong> {config.get('performance_requirements', 'standard').title()}</p>
            <p><strong>Environment:</strong> {config.get('environment', 'unknown').title()}</p>
            <p><strong>Number of Agents:</strong> {config.get('number_of_agents', 1)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # RDS vs EC2 Comparison
    st.markdown("**⚖️ RDS vs EC2 Deployment Comparison:**")
    
    comparison_col1, comparison_col2 = st.columns(2)
    
    with comparison_col1:
        # Create comparison chart
        comparison_data = {
            'Criteria': ['Management', 'Cost', 'Performance', 'Scalability', 'Control', 'Total Score'],
            'RDS Score': [90, 80, 85, 90, 60, deployment_rec.get('rds_score', 0)],
            'EC2 Score': [60, 85, 90, 80, 95, deployment_rec.get('ec2_score', 0)]
        }
        
        fig_comparison = px.bar(
            comparison_data,
            x='Criteria',
            y=['RDS Score', 'EC2 Score'],
            title="RDS vs EC2 Scoring Breakdown",
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with comparison_col2:
        st.markdown("**📊 Deployment Decision Factors:**")
        
        primary_reasons = deployment_rec.get('primary_reasons', [])
        
        st.markdown(f"""
        <div class="deployment-comparison-card">
            <h4>Why {recommendation}?</h4>
            <ul>
                {"".join([f"<li>{reason}</li>" for reason in primary_reasons[:4]])}
            </ul>
            <p><strong>Confidence Level:</strong> {confidence*100:.1f}%</p>
            <p><strong>Decision Strength:</strong> {"Strong" if abs(rds_score - ec2_score) > 20 else "Moderate" if abs(rds_score - ec2_score) > 10 else "Weak"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reader/Writer Configuration
    st.markdown("**🔄 Reader/Writer Instance Configuration:**")
    
    reader_writer = aws_sizing.get('reader_writer_config', {})
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>📊 Instance Distribution</h4>
            <p><strong>Writer Instances:</strong> {reader_writer.get('writers', 1)}</p>
            <p><strong>Reader Instances:</strong> {reader_writer.get('readers', 0)}</p>
            <p><strong>Total Instances:</strong> {reader_writer.get('total_instances', 1)}</p>
            <p><strong>Read Capacity:</strong> {reader_writer.get('read_capacity_percent', 0):.1f}%</p>
            <p><strong>Write Capacity:</strong> {reader_writer.get('write_capacity_percent', 100):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with config_col2:
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🎯 Configuration Reasoning</h4>
            <p><strong>Database Size:</strong> {config.get('database_size_gb', 0):,} GB</p>
            <p><strong>Performance Requirement:</strong> {config.get('performance_requirements', 'standard').title()}</p>
            <p><strong>Environment:</strong> {config.get('environment', 'unknown').title()}</p>
            <p><strong>Recommended Read Split:</strong> {reader_writer.get('recommended_read_split', 0):.0f}%</p>
            <p><strong>AI Optimization:</strong> {reader_writer.get('ai_insights', {}).get('optimization_potential', '0%')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with config_col3:
        ai_insights_rw = reader_writer.get('ai_insights', {})
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🤖 AI Configuration Insights</h4>
            <p><strong>Complexity Impact:</strong> {ai_insights_rw.get('complexity_impact', 0):.0f}/10</p>
            <p><strong>Agent Scaling Impact:</strong> {ai_insights_rw.get('agent_scaling_impact', 1)} agents</p>
            <p><strong>Scaling Factors:</strong></p>
            <ul>
                {"".join([f"<li>{factor}</li>" for factor in ai_insights_rw.get('scaling_factors', ['Standard scaling'])[:2]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Recommendations
    st.markdown("**🤖 AI Performance Recommendations:**")
    
    if recommendation == 'RDS':
        ai_recommendations = aws_sizing.get('rds_recommendations', {}).get('ai_recommendations', [])
    else:
        ai_recommendations = aws_sizing.get('ec2_recommendations', {}).get('ai_recommendations', [])
    
    if ai_recommendations:
        for i, recommendation_text in enumerate(ai_recommendations, 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            
            st.markdown(f"""
            <div class="ai-recommendation-card">
                <h5>AI Recommendation {i}: {recommendation_text}</h5>
                <p><strong>Expected Impact:</strong> {impact}</p>
                <p><strong>Implementation Complexity:</strong> {complexity}</p>
                <p><strong>Priority:</strong> {"Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"}</p>
            </div>
            """)
    
    # Instance Comparison Table
    st.markdown("**📋 Available Instance Options:**")
    
    pricing_data = aws_sizing.get('pricing_data', {})
    
    if recommendation == 'RDS':
        instances = pricing_data.get('rds_instances', {})
    else:
        instances = pricing_data.get('ec2_instances', {})
    
    if instances:
        instance_data = []
        current_instance = aws_sizing.get(f'{recommendation.lower()}_recommendations', {}).get('primary_instance', '')
        
        for instance_type, specs in instances.items():
            instance_data.append({
                'Instance Type': instance_type,
                'vCPU': specs.get('vcpu', 'N/A'),
                'Memory (GB)': specs.get('memory', 'N/A'),
                'Cost per Hour': f"${specs.get('cost_per_hour', 0):.4f}",
                'Monthly Cost': f"${specs.get('cost_per_hour', 0) * 24 * 30:.0f}",
                'Selected': '✅' if instance_type == current_instance else ''
            })
        
        df_instances = pd.DataFrame(instance_data)
        st.dataframe(df_instances, use_container_width=True)
    
    # Cost Optimization Suggestions
    st.markdown("**💡 Cost Optimization Suggestions:**")
    
    opt_col1, opt_col2 = st.columns(2)
    
    with opt_col1:
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>💰 Cost Reduction Strategies</h4>
            <ul>
                <li>Consider Reserved Instances for 20-30% savings</li>
                <li>Use Spot Instances for non-production workloads</li>
                <li>Implement automated scaling policies</li>
                <li>Optimize storage type selection</li>
                <li>Enable CloudWatch detailed monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with opt_col2:
        st.markdown("""
        <div class="detailed-analysis-section">
            <h4>📈 Performance Optimization</h4>
            <ul>
                <li>Configure read replicas for read scaling</li>
                <li>Enable enhanced monitoring</li>
                <li>Optimize database parameters</li>
                <li>Implement connection pooling</li>
                <li>Consider caching strategies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    
    # Professional footer with FSx capabilities
    st.markdown("""
    <div class="enterprise-footer">
        <h4>🚀 AWS Enterprise Database Migration Analyzer AI v3.0</h4>
        <p>Powered by Anthropic Claude AI • Real-time AWS Integration • Professional Migration Analysis • Advanced Agent Scaling • FSx Destination Analysis</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            🔬 Advanced Network Intelligence • 🎯 AI-Driven Recommendations • 📊 Executive Reporting • 🤖 Multi-Agent Optimization • 🗄️ S3/FSx Comparisons
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())




