# -----  ByteFF: ByteDance Force Field -----
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# This file is modified from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/basic_gnn.py.
# Copyright (c) 2023 PyG Team <team@pyg.org>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
import math
import typing as T

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear, ModuleList
from torch_geometric.nn.conv import (GATConv, GATv2Conv, GINConv, GINEConv,
                                     MessagePassing)
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (activation_resolver,
                                         normalization_resolver)
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import scatter, softmax
from torch_geometric.utils._trim_to_layer import TrimToLayer


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    supports_edge_attr: bool
    supports_edge_update: bool

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: T.Optional[int] = None,
        dropout: float = 0.0,
        act: T.Union[str, T.Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: T.Optional[T.Dict[str, T.Any]] = None,
        norm: T.Union[str, T.Callable, None] = None,
        norm_kwargs: T.Optional[T.Dict[str, T.Any]] = None,
        jk: T.Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                hidden_channels,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

        # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        self._trim = TrimToLayer()

    def init_conv(self, in_channels: T.Union[int, T.Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        x_batch: Tensor = None,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            x_batch: Tensor : The batch vector, which assigns each node to a specific example.
        """

        xs: T.List[Tensor] = []
        for i in range(self.num_layers):

            if self.supports_edge_update:
                x, edge_attr = self.convs[i](x,
                                             edge_index,
                                             edge_attr=edge_attr)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x, x_batch)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x

        if self.supports_edge_update:
            return x, edge_attr
        else:
            return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_update = False
    supports_edge_attr = False

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)


class GINE(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_update = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP([in_channels, out_channels, out_channels],
                  act=self.act,
                  act_first=self.act_first,
                  norm=None)
        return GINEConv(mlp, **kwargs)


class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_update = False
    supports_edge_attr = True

    def init_conv(self, in_channels: T.Union[int, T.Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels,
                    out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=self.dropout,
                    **kwargs)


class GTConv(MessagePassing):  # pylint: disable=abstract-method
    r"""Modified from torch_geometric.nn.TransformerConv.
    local version of EGT and remove edge part. https://arxiv.org/abs/2108.03348
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        at_channels: int = 16,
        dropout: float = 0.,
        ffn_dims: T.Optional[list[int]] = None,
        scale_by_degree: bool = False,
        act: str = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.at_channels = at_channels
        self.heads = heads
        self.dropout = dropout
        self.scale_by_degree = scale_by_degree

        self.lin_key = Linear(in_channels, heads * at_channels)
        self.lin_query = Linear(in_channels, heads * at_channels)
        self.lin_value = Linear(in_channels, heads * at_channels)
        self.lin_proj = Linear(heads * at_channels, in_channels)

        self.norm1 = LayerNorm(in_channels)
        # self.norm2 = LayerNorm(in_channels)

        self.ffn = None
        if ffn_dims is not None:
            self.ffn = MLP(in_channels=in_channels,
                           hidden_channels=ffn_dims[0],
                           out_channels=out_channels,
                           num_layers=ffn_dims[1],
                           act=act,
                           norm=None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_proj.reset_parameters()
        self.norm1.reset_parameters()
        if self.ffn:
            self.ffn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj):
        r"""Runs the forward pass of the module. """

        H, C = self.heads, self.at_channels

        x0 = x.clone()
        x = self.norm1(x)
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # multi-head attention
        x = self.propagate(edge_index,
                           query=query,
                           key=key,
                           value=value,
                           size=None)
        x = x.view(-1, H * C)
        x = self.lin_proj(x) + x0

        x1 = x.clone()
        # x = self.norm2(x)
        x = self.ffn(x)
        out = x + x1

        return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: T.Optional[int]) -> Tensor:

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(
            self.at_channels)  # [nedge, head]
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if self.scale_by_degree:
            degree = scatter(torch.ones_like(index), index, 0, reduce='sum')
            degree = degree.index_select(0, index)
            alpha *= degree.to(alpha.dtype).unsqueeze(-1)

        out = value_j  # [nedge, head, at_channels]
        out = out * alpha.view(-1, self.heads, 1)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class EGTConv(MessagePassing):  # pylint: disable=abstract-method
    r"""Modified from torch_geometric.nn.TransformerConv.
    local version of EGT. https://arxiv.org/abs/2108.03348
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        at_channels: int = 16,
        dropout: float = 0.,
        ffn_dims: T.Optional[list[int]] = None,
        scale_by_degree: bool = False,
        act: str = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.at_channels = at_channels
        self.heads = heads
        self.dropout = dropout
        self.scale_by_degree = scale_by_degree

        self.lin_key = Linear(in_channels, heads * at_channels)
        self.lin_query = Linear(in_channels, heads * at_channels)
        self.lin_value = Linear(in_channels, heads * at_channels)
        self.lin_proj = Linear(heads * at_channels, in_channels)
        self.norm1 = LayerNorm(in_channels)
        self.norm2 = LayerNorm(in_channels)

        self.lin_gate = Linear(in_channels, heads)
        self.lin_edge = Linear(in_channels, heads)
        self.lin_proj_edge = Linear(heads, in_channels)
        self.norm1_edge = LayerNorm(in_channels)
        self.norm2_edge = LayerNorm(in_channels)

        self.ffn = None
        self.ffn_edge = None
        if ffn_dims is not None:
            self.ffn = MLP(in_channels=in_channels,
                           hidden_channels=ffn_dims[0],
                           out_channels=out_channels,
                           num_layers=ffn_dims[1],
                           act=act,
                           norm=None)
            self.ffn_edge = MLP(in_channels=in_channels,
                                hidden_channels=ffn_dims[0],
                                out_channels=out_channels,
                                num_layers=ffn_dims[1],
                                act=act,
                                norm=None)

        self._edge_alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_proj.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.lin_gate.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_proj_edge.reset_parameters()
        self.norm1_edge.reset_parameters()
        self.norm2_edge.reset_parameters()
        if self.ffn:
            self.ffn.reset_parameters()
            self.ffn_edge.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor):
        r"""Runs the forward pass of the module. """

        H, C = self.heads, self.at_channels

        x0 = x.clone()
        x = self.norm1(x)
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        edge_attr0 = edge_attr.clone()
        edge_attr = self.norm1_edge(edge_attr)
        gate = torch.sigmoid(self.lin_gate(edge_attr))
        edge_embedding = self.lin_edge(edge_attr)

        # multi-head attention
        x = self.propagate(edge_index,
                           query=query,
                           key=key,
                           value=value,
                           size=None,
                           gate=gate,
                           edge_embedding=edge_embedding)
        x = x.view(-1, H * C)
        x = self.lin_proj(x) + x0
        x1 = x.clone()
        x = self.norm2(x)
        x = self.ffn(x)
        out = x + x1

        edge_alpha = self._edge_alpha
        self._edge_alpha = None
        edge_alpha = self.lin_proj_edge(edge_alpha)
        edge_attr0 += edge_alpha
        edge_attr1 = edge_attr0.clone()
        edge_attr = self.ffn_edge(self.norm2_edge(edge_attr0)) + edge_attr1
        return out, edge_attr

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                index: Tensor, ptr: OptTensor, size_i: T.Optional[int],
                gate: Tensor, edge_embedding: Tensor) -> Tensor:

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(
            self.at_channels)  # [nedge, head]
        alpha += edge_embedding
        self._edge_alpha = alpha.clone()
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if self.scale_by_degree:
            degree = scatter(torch.ones_like(index), index, 0, reduce='sum')
            degree = degree.index_select(0, index)
            alpha *= degree.to(alpha.dtype).unsqueeze(-1)

        alpha = (gate * alpha).view(-1, self.heads, 1)  # [nedge, head, 1]
        out = value_j * alpha

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GT(BasicGNN):
    r"""Modified from torch_geometric.nn.TransformerConv.
    local version of EGT and edge part. https://arxiv.org/abs/2108.03348
    """

    supports_edge_attr = False
    supports_edge_update = False

    def init_conv(self, in_channels: T.Union[int, T.Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        assert self.norm is None, 'Layernorm has already added in GTConv'
        return GTConv(in_channels=in_channels,
                      out_channels=out_channels,
                      dropout=self.dropout,
                      act=self.act,
                      **kwargs)


class EGT(BasicGNN):
    r"""Modified from torch_geometric.nn.TransformerConv.
    local version of EGT. https://arxiv.org/abs/2108.03348
    """

    supports_edge_attr = True
    supports_edge_update = True

    def init_conv(self, in_channels: T.Union[int, T.Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        assert self.norm is None, 'Layernorm has already added in GTConv'
        return EGTConv(in_channels=in_channels,
                       out_channels=out_channels,
                       dropout=self.dropout,
                       act=self.act,
                       **kwargs)
