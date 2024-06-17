# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import time 
import pytorch_lightning as pl
import nugraph as ng
import torch
import numpy as np
from torch import nn
from torch_geometric.data import HeteroData, Batch
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class NuGraph3_model(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(NuGraph3_model, self).__init__()
        self.MODEL = ng.models.nugraph3.nugraph3.NuGraph3
        self.model = self.MODEL.load_from_checkpoint("gnn_models/nugraph3/1/hierarchical.ckpt", map_location='cpu')
        self.accelerator, self.devices = ng.util.configure_device()
        self.trainer = pl.Trainer(accelerator=self.accelerator, devices=self.devices,
                            logger=False)

    def forward(self, sp_num_nodes, u_x_dict, v_x_dict, y_x_dict, evt_num_nodes, \
                u_plane_u, u_nexus_sp, v_plane_v, v_nexus_sp, y_plane_y, y_nexus_sp, \
                u_in_evt, evt_owns_u, v_in_evt, evt_owns_v, y_in_evt, evt_owns_y, \
                sp_in_evt, evt_owns_sp, sp_nexus_u, sp_nexus_v, sp_nexus_y):

        # sp_num_nodes = torch.tensor(sp_num_nodes)
        # evt_num_nodes = torch.tensor(evt_num_nodes)

        # u_x_dict = torch.tensor(u_x_dict)
        # v_x_dict = torch.tensor(v_x_dict)
        # y_x_dict = torch.tensor(y_x_dict)

        # u_plane_u = torch.tensor(u_plane_u)
        # u_nexus_sp = torch.tensor(u_nexus_sp)
        # v_plane_v = torch.tensor(v_plane_v)
        # v_nexus_sp = torch.tensor(v_nexus_sp)
        # y_plane_y = torch.tensor(y_plane_y)
        # y_nexus_sp = torch.tensor(y_nexus_sp)

        # u_in_evt = torch.tensor(u_in_evt)
        # evt_owns_u = torch.tensor(evt_owns_u)
        # v_in_evt = torch.tensor(v_in_evt)
        # evt_owns_v = torch.tensor(evt_owns_v)
        # y_in_evt = torch.tensor(y_in_evt)
        # evt_owns_y = torch.tensor(evt_owns_y)

        # sp_in_evt = torch.tensor(sp_in_evt)
        # evt_owns_sp = torch.tensor(evt_owns_sp)

        # sp_nexus_u = torch.tensor(sp_nexus_u)
        # sp_nexus_v = torch.tensor(sp_nexus_v)
        # sp_nexus_y = torch.tensor(sp_nexus_y)

        hetero_batch = self.create_heteroBatch(sp_num_nodes, u_x_dict, v_x_dict, y_x_dict, evt_num_nodes, \
                u_plane_u, u_nexus_sp, v_plane_v, v_nexus_sp, y_plane_y, y_nexus_sp, \
                u_in_evt, evt_owns_u, v_in_evt, evt_owns_v, y_in_evt, evt_owns_y, \
                sp_in_evt, evt_owns_sp, sp_nexus_u, sp_nexus_v, sp_nexus_y)
        
        _, _, _, x = self.trainer.predict(self.model, hetero_batch)
        print(x)
        exit()
        return x 
    
    def create_heteroBatch(self, sp_num_nodes, u_x_dict, v_x_dict, y_x_dict, evt_num_nodes, \
                u_plane_u, u_nexus_sp, v_plane_v, v_nexus_sp, y_plane_y, y_nexus_sp, \
                u_in_evt, evt_owns_u, v_in_evt, evt_owns_v, y_in_evt, evt_owns_y, \
                sp_in_evt, evt_owns_sp, sp_nexus_u, sp_nexus_v, sp_nexus_y):
        
        output_dict = {
            "sp":
                {"num_nodes":sp_num_nodes},
            "u":
                {"x"},
            "v":
                {"x"},
            "y":
                {"x"},
            "evt":
                {"num_nodes":evt_num_nodes},
            ("u", "plane", "u"):
                {"edge_index":u_plane_u},
            ("u", "nexus", "sp"):
                {"edge_index":u_nexus_sp},
            ("v", "plane", "v"):
                {"edge_index":v_plane_v},
            ("v", "nexus", "sp"):
                {"edge_index":v_nexus_sp},
            ("y", "plane", "y"):
                {"edge_index":y_plane_y},
            ("y", "nexus", "sp"):
                {"edge_index":y_nexus_sp},
            ("u", "in", "evt"):
                {"edge_index":u_in_evt},
            ("evt", "owns", "u"):
                {"edge_index":evt_owns_u},
            ("v", "in", "evt"):
                {"edge_index":v_in_evt},
            ("evt", "owns", "v"):
                {"edge_index":evt_owns_v},
            ("y", "in", "evt"):
                {"edge_index":y_in_evt},
            ("evt", "owns", "y"):
                {"edge_index":evt_owns_y},
            ("sp", "in", "evt"):
                {"edge_index":sp_in_evt},
            ("evt", "owns", "sp"):
                {"edge_index":evt_owns_sp},
            ("sp", "nexus", "u"):
                {"edge_index":sp_nexus_u},
            ("sp", "nexus", "v"):
                {"edge_index":sp_nexus_v},
            ("sp", "nexus", "y"):
                {"edge_index":sp_nexus_y}         

        }
    
        # Create HeteroData object & HeteroBatch
        hetero_data = HeteroData(output_dict)

        hetero_batch = Batch.from_data_list([hetero_data])
        return hetero_batch

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get ouptput configuration
        e_evt_config = pb_utils.get_output_config_by_name(model_config, "e_evt")

        # Convert Triton types to numpy types
        self.e_evt_dtype = pb_utils.triton_string_to_numpy(
            e_evt_config["data_type"]
        )
        # Instantiate the PyTorch model
        self.NuGraph3_model = NuGraph3_model()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        e_evt_dtype = self.softmax0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get all inputs
            sp_num_nodes = pb_utils.get_input_tensor_by_name(request, "sp_num_nodes")
            evt_num_nodes = pb_utils.get_input_tensor_by_name(request, "evt_num_nodes")

            u_x_dict = pb_utils.get_input_tensor_by_name(request, "u_x_dict")
            v_x_dict = pb_utils.get_input_tensor_by_name(request, "v_x_dict")
            y_x_dict = pb_utils.get_input_tensor_by_name(request, "y_x_dict")

            u_plane_u = pb_utils.get_input_tensor_by_name(request, "u_plane_u")
            u_nexus_sp = pb_utils.get_input_tensor_by_name(request, "u_nexus_sp")
            v_plane_v = pb_utils.get_input_tensor_by_name(request, "v_plane_v")
            v_nexus_sp = pb_utils.get_input_tensor_by_name(request, "y_nexus_sp")
            y_plane_y = pb_utils.get_input_tensor_by_name(request, "y_plane_y")
            y_nexus_sp = pb_utils.get_input_tensor_by_name(request, "y_nexuss_sp")

            u_in_evt = pb_utils.get_input_tensor_by_name(request, "u_in_evt")
            evt_owns_u = pb_utils.get_input_tensor_by_name(request, "evt_owns_u")
            v_in_evt = pb_utils.get_input_tensor_by_name(request, "v_in_evt")
            evt_owns_v = pb_utils.get_input_tensor_by_name(request, "evt_owns_v")
            y_in_evt = pb_utils.get_input_tensor_by_name(request, "y_in_evt")
            evt_owns_y = pb_utils.get_input_tensor_by_name(request, "evt_owns_y")

            sp_in_evt = pb_utils.get_input_tensor_by_name(request, "sp_in_evt")
            evt_owns_sp = pb_utils.get_input_tensor_by_name(request, "evt_owns_sp")

            sp_nexus_u = pb_utils.get_input_tensor_by_name(request, "sp_nexus_u")
            sp_nexus_v = pb_utils.get_input_tensor_by_name(request, "sp_nexus_v")
            sp_nexus_y = pb_utils.get_input_tensor_by_name(request, "sp_nexus_y")


            output = self.NuGraph3_model(sp_num_nodes, u_x_dict, v_x_dict, y_x_dict, evt_num_nodes, \
                u_plane_u, u_nexus_sp, v_plane_v, v_nexus_sp, y_plane_y, y_nexus_sp, \
                u_in_evt, evt_owns_u, v_in_evt, evt_owns_v, y_in_evt, evt_owns_y, \
                sp_in_evt, evt_owns_sp, sp_nexus_u, sp_nexus_v, sp_nexus_y)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("e_evt", output.astype(e_evt_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")