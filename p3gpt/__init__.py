"""
P3GPT - Precious3 GPT Package

A Python package for multimodal language models focused on biomedical applications.
"""

__version__ = "0.1.0"

# Import key components for easier access
from p3gpt.handlers.p3_multimodal import Precious3MptModel, Precious3MPTForCausalLM
from p3gpt.handlers.p3_multimodal_handler import BaseHandler, EndpointHandler, SMILESHandler, HandlerFactory
