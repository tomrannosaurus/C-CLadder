import argparse
from copy import deepcopy
import datetime
import random
import uuid
import networkx as nx
from enum import Enum
import json
import os

class CausalGraphType(Enum):
    CONFOUNDING = "confounding"
    MEDIATION = "mediation"
    FORK = "fork"
    COLLISION = "collision"
    DIAMOND = "diamond"
    DIAMONDCUT = "diamondcut"
    CHAIN = "chain"
    IV = "IV"
    ARROWHEAD = "arrowhead"
    FRONTDOOR = "frontdoor"
    NONE = "none"


class CausalGraphNodeType(Enum):
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    OTHER = "other"


class CorruptionType(Enum):
    REVERSE_RANDOM_EDGE = "reverse_random_edge"
    ADD_COLLIDER = "add_collider"
    ADD_CONFOUNDER = "add_confounder"
    ADD_MEDIATOR = "add_mediator"
    NONE = "none"


class PromptType(Enum):
    NO_GRAPH = "no_graph"
    ORIGINAL_GRAPH = "original_graph"
    CORRUPTED_GRAPH = "corrupted_graph"

class CladderCategory(Enum):
    COMMONSENSE = "commonsense"
    ANTICOMMONSENSE = "anticommonsense"
    NONSENSE = "nonsense"

class CausalGraphNode:
    name: str
    is_observed: bool | None = True
    node_type: CausalGraphNodeType = CausalGraphNodeType.OTHER

    def __init__(
        self,
        name: str,
        is_observed: bool | None = None,
        node_type: CausalGraphNodeType = CausalGraphNodeType.OTHER,
    ) -> None:
        self.name = name
        self.is_observed = is_observed
        self.node_type = node_type

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def capitalize(self) -> str:
        return self.name.capitalize()


class CausalGraph(nx.DiGraph):
    type: CausalGraphType = CausalGraphType.NONE
    corruption_type: CorruptionType = CorruptionType.NONE

    def __init__(self, type: CausalGraphType = CausalGraphType.NONE) -> None:
        self.type = type

        match type:
            case CausalGraphType.CONFOUNDING:
                edges = [
                    (CausalGraphNode("V1"), CausalGraphNode("X")),
                    (CausalGraphNode("V1"), CausalGraphNode("Y")),
                    (CausalGraphNode("X"), CausalGraphNode("Y")),
                ]
            case CausalGraphType.MEDIATION:
                edges = [
                    (CausalGraphNode("X"), CausalGraphNode("V2")),
                    (CausalGraphNode("V2"), CausalGraphNode("Y")),
                    (CausalGraphNode("X"), CausalGraphNode("Y")),
                ]
            case CausalGraphType.FORK:
                edges = [
                    (CausalGraphNode("X"), CausalGraphNode("Y")),
                    (CausalGraphNode("V2"), CausalGraphNode("Y")),
                ]
            case CausalGraphType.COLLISION:
                edges = [
                    (CausalGraphNode("X"), CausalGraphNode("Y")),
                    (CausalGraphNode("X"), CausalGraphNode("V3")),
                    (CausalGraphNode("Y"), CausalGraphNode("V3")),
                ]
            case CausalGraphType.DIAMOND:
                edges = [
                    (CausalGraphNode("X"), CausalGraphNode("V3")),
                    (CausalGraphNode("X"), CausalGraphNode("V2")),
                    (CausalGraphNode("V2"), CausalGraphNode("Y")),
                    (CausalGraphNode("V3"), CausalGraphNode("Y")),
                ]
            case CausalGraphType.DIAMONDCUT:
                edges = [
                    (CausalGraphNode("V1"), CausalGraphNode("V3")),
                    (CausalGraphNode("V1"), CausalGraphNode("X")),
                    (CausalGraphNode("X"), CausalGraphNode("Y")),
                    (CausalGraphNode("V3"), CausalGraphNode("Y")),
                ]
            case CausalGraphType.CHAIN:
                edges = [
                    (CausalGraphNode("X"), CausalGraphNode("V2")),
                    (CausalGraphNode("V2"), CausalGraphNode("Y")),
                ]
            case CausalGraphType.IV:
                edges = [
                    (CausalGraphNode("V1"), CausalGraphNode("X")),
                    (CausalGraphNode("V2"), CausalGraphNode("X")),
                    (CausalGraphNode("V1"), CausalGraphNode("Y")),
                    (CausalGraphNode("X"), CausalGraphNode("Y")),
                ]
            case CausalGraphType.ARROWHEAD:
                edges = [
                    (CausalGraphNode("X"), CausalGraphNode("V3")),
                    (CausalGraphNode("V2"), CausalGraphNode("V3")),
                    (CausalGraphNode("X"), CausalGraphNode("Y")),
                    (CausalGraphNode("V2"), CausalGraphNode("Y")),
                    (CausalGraphNode("V3"), CausalGraphNode("Y")),
                ]
            case CausalGraphType.FRONTDOOR:
                edges = [
                    (CausalGraphNode("V1"), CausalGraphNode("X")),
                    (CausalGraphNode("X"), CausalGraphNode("V3")),
                    (CausalGraphNode("V1"), CausalGraphNode("Y")),
                    (CausalGraphNode("V3"), CausalGraphNode("Y")),
                ]
            case _:
                edges = []

        super().__init__(edges)
        self._define_node_types()

    def _define_node_types(self) -> None:
        # Define treatment and outcome nodes based on common conventions
        for node in self.nodes:
            if node.name == "X":
                node.node_type = CausalGraphNodeType.TREATMENT
            elif node.name == "Y":
                node.node_type = CausalGraphNodeType.OUTCOME

    def relabel_nodes(self, relabelling_dict: dict) -> None:
        # Extract relabelling mapping from dicts like this:
        # variable_mapping": {
        #     "Xname": "husband",
        #     "X1": "alarm set by husband",
        #     "X0": "alarm not set by husband",
        #     "V2name": "wife",
        #     "V21": "alarm set by wife",
        #     "V20": "alarm not set by wife",
        #     "Yname": "alarm clock",
        #     "Y1": "ringing alarm",
        #     "Y0": "silent alarm"
        # }
        mapping = {}
        for key, value in relabelling_dict.items():
            if key.endswith("name"):
                node_key = key[:-4]  # Remove 'name' suffix to get the node identifier
                for node in self.nodes:
                    if node.name == node_key:
                        mapping[CausalGraphNode(node_key)] = CausalGraphNode(
                            name=value,
                            is_observed=node.is_observed,
                            node_type=node.node_type,
                        )

        nx.relabel_nodes(self, mapping, copy=False)

    def set_observed_status(self, background: str) -> None:
        """
        If the text `NAME is unobserved.` is contained in the background,
        set the is_observed attribute of the corresponding node to False.
        Otherwise, set it to True.
        """
        for node in self.nodes:
            if f"{node.capitalize()} is unobserved." in background:
                node.is_observed = False
            else:
                node.is_observed = True

    def get_node_by_type(
        self, node_type: CausalGraphNodeType
    ) -> CausalGraphNode | None:
        for node in self.nodes:
            if node.node_type == node_type:
                return node
        print(f"Warning: No node of type {node_type} found in graph.")
        return None

    # Corrupt graph in place
    def corrupt(self, corruption_type: CorruptionType) -> None:
        self.corruption_type = corruption_type
        match corruption_type:
            case CorruptionType.REVERSE_RANDOM_EDGE:
                if self.edges:
                    random_edge = random.choice(list(self.edges))
                    self.remove_edge(*random_edge)
                    self.add_edge(random_edge[1], random_edge[0])
            case CorruptionType.ADD_COLLIDER:
                new_node = CausalGraphNode(f"collider{len(self.nodes)+1}")
                if self.nodes:
                    # Connect to treatment and outcome node if they exist.
                    treatment_node = self.get_node_by_type(
                        CausalGraphNodeType.TREATMENT
                    )
                    outcome_node = self.get_node_by_type(CausalGraphNodeType.OUTCOME)
                    if treatment_node and outcome_node:
                        self.add_node(new_node)
                        self.add_edge(treatment_node, new_node)
                        self.add_edge(outcome_node, new_node)
            case CorruptionType.ADD_CONFOUNDER:
                new_node = CausalGraphNode(f"confounder{len(self.nodes)+1}")
                if self.nodes:
                    # Connect to treatment and outcome node if they exist.
                    treatment_node = self.get_node_by_type(
                        CausalGraphNodeType.TREATMENT
                    )
                    outcome_node = self.get_node_by_type(CausalGraphNodeType.OUTCOME)
                    if treatment_node and outcome_node:
                        self.add_node(new_node)
                        self.add_edge(new_node, treatment_node)
                        self.add_edge(new_node, outcome_node)
            case CorruptionType.ADD_MEDIATOR:
                new_node = CausalGraphNode(f"mediator{len(self.nodes)+1}")
                if self.nodes:
                    # Connect to treatment and outcome node if they exist.
                    treatment_node = self.get_node_by_type(
                        CausalGraphNodeType.TREATMENT
                    )
                    outcome_node = self.get_node_by_type(CausalGraphNodeType.OUTCOME)
                    if treatment_node and outcome_node:
                        self.add_node(new_node)
                        self.add_edge(treatment_node, new_node)
                        self.add_edge(new_node, outcome_node)

    def to_graphml(self) -> str:
        return "\n".join(str(item) for item in nx.generate_graphml(self))

    def verbalize(self) -> str:
        """
        Dynamically generate background text for the graph.
        - For each edge: '{parent} has a direct effect on {children}.'
        - For each unobserved node: '{vname} is unobserved.'
        - For each unconnected node: '{vname} is unconnected.'
        Uses node names directly.
        """
        background_lines = []
        # Edges
        for node in self.nodes:
            if self.out_degree(node) == 0:  # No children
                continue
            background_line = f"{node.capitalize()} has a direct effect on "
            successors = list(self.successors(node))
            for i, child in enumerate(successors):
                background_line += f"{child}"
                if i == len(successors) - 1:
                    background_line += "."
                else:
                    background_line += " and "
            background_lines.append(background_line)

        # Unobserved nodes
        for node in self.nodes:
            if not node.is_observed and node.is_observed is not None:
                background_lines.append(f"{node.capitalize()} is unobserved.")

        # Unconnected nodes (no in-edges and no out-edges)
        for node in self.nodes:
            if self.in_degree(node) == 0 and self.out_degree(node) == 0:
                background_lines.append(f"{node.capitalize()} is unconnected.")

        return " ".join(background_lines)


class Prompt:
    question_id: int
    model_id: int
    type: PromptType = PromptType.NO_GRAPH
    graph: CausalGraph | None = None
    original_background: str
    given_info: str
    question: str
    ground_truth: str
    category: CladderCategory
    rung: int

    def __init__(
        self,
        question_id: int,
        model_id: int,
        original_background: str,
        given_info: str,
        question: str,
        ground_truth: str,
        category: CladderCategory,
        rung: int,
        graph: CausalGraph | None = None,
    ):
        self.question_id = question_id
        self.model_id = model_id
        self.graph = graph
        self.original_background = original_background
        self.given_info = given_info
        self.question = question
        self.ground_truth = ground_truth
        if graph is None:
            self.type = PromptType.NO_GRAPH
        elif graph.corruption_type == CorruptionType.NONE:
            self.type = PromptType.ORIGINAL_GRAPH
        else:
            self.type = PromptType.CORRUPTED_GRAPH
        self.category = category
        self.rung = rung

    def background(self) -> str:
        background = "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: "
        if self.graph is not None:
            background += self.graph.verbalize()
        return background

    def to_string(self) -> str:
        prompt_str = f"{self.background()}\n\n{self.given_info}\n\n{self.question}"
        prompt_str += '\n\nBased on all the reasoning above, output one word to answer the initial question with just "Yes" or "No".'
        return prompt_str


def generate_prompt_variations(prompt: Prompt) -> list[Prompt]:
    variations = []

    if prompt.graph is None:
        print(
            f"Warning: Prompt {prompt.question_id} has no graph to generate variations from."
        )
        return [prompt]

    # Original prompt with no graph
    variations.append(
        Prompt(
            question_id=prompt.question_id,
            model_id=prompt.model_id,
            original_background=prompt.original_background,
            given_info=prompt.given_info,
            question=prompt.question,
            graph=None,
            ground_truth=prompt.ground_truth,
            category=prompt.category,
            rung=prompt.rung,
        )
    )

    # Original prompt with original graph
    if prompt.graph is not None:
        variations.append(
            Prompt(
                question_id=prompt.question_id,
                model_id=prompt.model_id,
                original_background=prompt.original_background,
                given_info=prompt.given_info,
                question=prompt.question,
                graph=prompt.graph,
                ground_truth=prompt.ground_truth,
                category=prompt.category,
                rung=prompt.rung,
            )
        )

        # Prompts with corrupted graphs
        for corruption in CorruptionType:
            if corruption == CorruptionType.NONE:
                continue
            corrupted_graph = deepcopy(prompt.graph)
            corrupted_graph.corrupt(corruption)
            variations.append(
                Prompt(
                    question_id=prompt.question_id,
                    model_id=prompt.model_id,
                    original_background=prompt.original_background,
                    given_info=prompt.given_info,
                    question=prompt.question,
                    graph=corrupted_graph,
                    ground_truth=prompt.ground_truth,
                    category=prompt.category,
                    rung=prompt.rung,
                )
            )

    return variations


def get_prompts_from_json(question_file, models_file) -> list[list[Prompt]]:
    data = json.load(question_file)
    models = json.load(models_file)
    prompts = []
    print(f"Total items in JSON: {len(data)}")
    for item in data:
        graph_type = CausalGraphType(item["meta"]["graph_id"])
        graph = CausalGraph(type=graph_type)
        model_id = item["meta"]["model_id"]
        # Search the models array for the matching model_id
        model = next((m for m in models if m["model_id"] == model_id), None)
        if model is None:
            print(f"Model with model_id {model_id} not found")
            continue

        graph.relabel_nodes(model["variable_mapping"])
        graph.set_observed_status(model["background"])

        default_prompt = Prompt(
            question_id=item["question_id"],
            model_id=model_id,
            original_background=model["background"],
            given_info=item["given_info"],
            question=item["question"],
            graph=graph,
            ground_truth=item.get("answer"),
            category=CladderCategory(item["sense"]),
            rung=item["meta"]["rung"],
        )

        prompts.append(generate_prompt_variations(default_prompt))

    return prompts

def generate_dataset(prompts, output_file: str):
    print("Generating dataset...")
    counter = 0
    total_prompts = sum(len(prompt) for prompt in prompts)
    for prompt in prompts:
        for variation in prompt:
            # Progress output
            counter += 1
            progress = (counter / total_prompts) * 100
            print(
                f"\rGenerating prompt {counter}/{total_prompts}: {progress:.2f}% complete", end=""
            )
            dataset_entry = {
                "uuid": uuid.uuid4().hex,
                "cladder_question_id": variation.question_id,
                "cladder_model_id": variation.model_id,
                "cladder_ground_truth": variation.ground_truth,
                "cladder_category": variation.category.value,
                "cladder_rung": variation.rung,
                "prompt_type": variation.type.value,
                "graph": (
                    {
                        "type": variation.graph.type.value,
                        "graphml": variation.graph.to_graphml(),
                        "corruption_type": variation.graph.corruption_type.value,
                    }
                    if variation.graph
                    else None
                ),
                "prompt": variation.to_string(),
            }
            # Append to JSON file
            with open(output_file, "a") as f:
                f.write(json.dumps(dataset_entry) + "\n")
    print(f"\nGenerated dataset with {counter} entries.")


def main():
    dirname = os.path.dirname(__file__)
    meta_file = os.path.join(dirname, "./cladder-v1-meta-models.json")

    parser = argparse.ArgumentParser(
        description="Generate corrupted causal graphs and corresponding prompts."
    )

    parser.add_argument(
        "--meta_file",
        type=str,
        default=meta_file,
        help="Path to the meta models JSON file.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="corrupted_causal_graphs_dataset.jsonl",
        help="Path to the output JSONL file which contains the models responses. Default: corrupted_causal_graphs_dataset.jsonl",
    )

    parser.add_argument(
        "question_file", type=str, help="Path to the questions JSON file."
    )

    args = parser.parse_args()
    meta_file = args.meta_file
    question_file_path = args.question_file
    output_file = args.output_file

    if os.path.exists(output_file):
        print(
            f"Output file {output_file} already exists. Please remove it or choose a different name."
        )
        return

    # Import JSON and generate graphs
    with (
        open(question_file_path, "r") as question_file,
        open(meta_file, "r") as models_file,
    ):
        prompts = get_prompts_from_json(question_file, models_file)

    print(f"Loaded {len(prompts)} prompts from JSON.")

    generate_dataset(prompts, output_file)

if __name__ == "__main__":
    main()
