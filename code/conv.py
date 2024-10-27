import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple
import numpy as np

def get_conv_weight_and_bias(
        filter_size: Tuple[int, int],
        num_groups: int,
        input_channels: int,
        output_channels: int,
        bias: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    # assert that num_filters is divisible by num_groups
    assert input_channels % num_groups == 0, "input channels must be divisible by groups number"
    # assert that num_channels is divisible by num_groups
    assert output_channels % num_groups == 0, "output channels must be divisible by groups number"
    input_channels_per_group = input_channels // num_groups

    # initialize the weight matrix
    #weight_matrix = torch.randn(input_channels, output_channels, *filter_size) # Order was wrong
    weight_matrix = torch.randn(output_channels, input_channels_per_group, *filter_size)
    # initialize the bias vector
    if bias:
        bias_vector = torch.ones(output_channels)
    else:
        bias_vector = None
    return weight_matrix, bias_vector


class MyConvStub:
    def __init__(
            self,
            kernel_size: Tuple[int, int],
            num_groups: int,
            input_channels: int,
            output_channels: int,
            bias: bool,
            stride: int,
            dilation: int,
            *args,  # Capture extra positional arguments
            padding: int = 0,  # Default value for padding
            **kwargs  # Capture extra keyword arguments
    ):
        self.weight, self.bias = get_conv_weight_and_bias(kernel_size, num_groups, input_channels, output_channels, bias)
        self.groups = num_groups
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def apply_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            # Apply padding (add zeros around the borders)
            padded_x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        else:
            padded_x = x
        return padded_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply padding before performing the convolution
        x = self.apply_padding(x)

        # Extract shape information from the input tensor
        batch_size, in_channels, input_height, input_width = x.shape
        
        # Extract shape information from the weight matrix
        out_channels, _, kernel_height, kernel_width = self.weight.shape

        # Calculate the number of input channels per group
        input_channels_per_group = in_channels // self.groups

        # Calculate output dimensions (assuming no padding for simplicity)
        output_height = (input_height - (kernel_height - 1) * self.dilation - 1) // self.stride + 1
        output_width = (input_width - (kernel_width - 1) * self.dilation - 1) // self.stride + 1

        # Initialize output tensor with zeros
        output = torch.zeros((batch_size, out_channels, output_height, output_width))

        # Perform grouped convolution manually
        for b in range(batch_size):  # Loop over each image in the batch
            for g in range(self.groups):  # Loop over each group
                # Calculate start and end index for the current group
                start_idx = g * input_channels_per_group
                end_idx = (g + 1) * input_channels_per_group

                # Slice the input and weights for this group
                input_slice = x[b, start_idx:end_idx, :, :]  # Slice input for the current group
                # Slice the weights for the current group
                weight_group_slice = self.weight[g * (out_channels // self.groups):(g + 1) * (out_channels // self.groups), :, :, :]

                for oc in range(out_channels // self.groups):  # Loop over output channels within the group
                    for h in range(output_height):  # Loop over output height
                        for w in range(output_width):  # Loop over output width
                            h_start = h * self.stride
                            w_start = w * self.stride

                            # Extract input patch considering dilation
                            input_patch = input_slice[:, 
                                                    h_start:h_start + kernel_height * self.dilation:self.dilation,
                                                    w_start:w_start + kernel_width * self.dilation:self.dilation]

                            # Apply the filter (dot product)
                            output[b, g * (out_channels // self.groups) + oc, h, w] += torch.sum(input_patch * weight_group_slice[oc, :, :, :])

        return output


class MyFilterStub:
    def __init__(
            self,
            filter: torch.Tensor,
            input_channels: int,
    ):
        self.weight = filter
        self.input_channels = input_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple implementation applying the filter channelwise
        output = torch.zeros_like(x)
        for i in range(self.input_channels):
            output[:, i, :, :] = torch.nn.functional.conv2d(
                x[:, i:i+1, :, :], self.weight.unsqueeze(0).unsqueeze(0), padding=1
            )
        return output
