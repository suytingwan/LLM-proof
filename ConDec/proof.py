"""
Proof steps and proof trees.
"""
from common import *
import itertools
import numpy as np
from copy import deepcopy


class InvalidProofStep(Exception):
    pass


class ProofStep:
    def __init__(self, proof: "Proof", s: str, strict: bool) -> None:
        self.proof = proof
        if s.count(" -> ") != 1:
            raise InvalidProofStep
        premises, conclusion = s.split(" -> ")

        self.premise_idents = []
        self.premise_sents = []
        for p in premises.split(" & "):
            if not re.fullmatch(r"(sent|int)\d+", p):
                # Ill-formatted premises.
                raise InvalidProofStep
            self.premise_idents.append(p)
            try:
                sent = proof.ident2sent(p)
                self.premise_sents.append(sent)
            except KeyError:
                # Unsatisfied premises.
                raise InvalidProofStep

        if conclusion == "hypothesis":
            self.conclusion_ident = "hypothesis"
            self.conclusion_sent = proof.hypothesis
        else:
            m = re.fullmatch(r"(?P<ident>int\d+): (?P<sent>.+)", conclusion)
            if m is None or re.search(r"(sent|int)\d+", m["sent"]):
                # Ill-formatted conclusion.
                raise InvalidProofStep
            self.conclusion_ident = m["ident"]
            self.conclusion_sent = m["sent"]
            if self.conclusion_sent == self.proof.hypothesis:
                # Intermediate conclusion identical with the hypothesis.
                raise InvalidProofStep
            if strict and (
                self.conclusion_sent in self.proof.context.values()
                or any(
                    self.conclusion_sent == step.conclusion_sent
                    for step in self.proof.proof_steps
                )
            ):
                # Intermediate conclusion identical with premises or an existing intermediate conclusion.
                raise InvalidProofStep

        if self.conclusion_ident in self.premise_idents:
            raise InvalidProofStep

    def __str__(self) -> str:
        if self.conclusion_ident == "hypothesis":
            return f"{' & '.join(self.premise_idents)} -> hypothesis"
        else:
            return f"{' & '.join(self.premise_idents)} -> {self.conclusion_ident}: {self.conclusion_sent}"

    def is_final(self) -> bool:
        return self.conclusion_ident == "hypothesis"


class Proof:
    def __init__(
        self,
        context: Union[str, OrderedDict[str, str]],
        hypothesis: str,
        proof_text: str,
        proof_id: int,
        strict: bool,
        requires_complete: bool = False,
    ) -> str:
        if isinstance(context, str):
            context = extract_context(context)
        self.hypothesis = hypothesis
        self.proof_id = proof_id
        self.strict = strict
        self.requires_complete = requires_complete

        proof_text = proof_text.strip()
        if proof_text.endswith(";"):
            proof_text = proof_text[:-1]

        self.context = context
        self.proof_text = proof_text

        self.proof_steps = []
        for s in proof_text.split(";"):
            s = s.strip()
            if s == "":
                continue
            self.proof_steps.append(ProofStep(self, s, strict))
        self.size = len(self.proof_steps)

        if requires_complete:
            assert self.is_complete()

    def __str__(self) -> str:
        return self.proof_text

    def __contains__(self, s: str) -> bool:
        return s in self.context.values() or any(
            s == step.conclusion_sent for step in self.proof_steps
        )

    def is_empty(self) -> bool:
        return self.proof_text == ""

    def serialize_context(self) -> str:
        return normalize(" ".join(f"{k}: {v}" for k, v in self.context.items()))

    def execute(self, step: ProofStep) -> None:
        assert not self.is_complete()
        assert step.proof is self
        if len(self.proof_steps) > 0:
            self.proof_text += "; "
        self.proof_text += str(step)
        self.proof_steps.append(step)

    def to_tree(self) -> Tree:
        return deserialize(self.hypothesis, self.context, self.proof_text)

    def ident2sent(self, ident: str) -> str:
        if ident == "hypothesis":
            return self.hypothesis
        elif re.fullmatch(r"sent\d+", ident):
            return self.context[ident]
        elif re.fullmatch(r"int\d+", ident):
            for step in self.proof_steps:
                if step.conclusion_ident == ident:
                    return step.conclusion_sent
            raise KeyError
        else:
            raise KeyError

    def shuffle_context(self) -> "Proof":
        """
        Randomly shuffle the identifiers of the supporting facts.
        """
        num_sents = len(self.context)
        permutation = list(range(num_sents))
        random.shuffle(permutation)
        inv_permutation = [permutation.index(i) for i in range(num_sents)]

        shuffled_context = " ".join(
            f"sent{i+1}: {self.context[f'sent{permutation[i]+1}']}"
            for i in range(num_sents)
        )

        tokens = []
        for t in self.proof_text.split():
            if re.fullmatch(r"sent\d+", t):
                i = int(t[4:])
                tokens.append(f"sent{inv_permutation[i-1]+1}")
            else:
                tokens.append(t)
        renamed_proof = " ".join(tokens)

        return Proof(
            shuffled_context,
            self.hypothesis,
            renamed_proof,
            self.strict,
            self.requires_complete,
        ), permutation, inv_permutation

    def back_proof(self, proof_text, permutation) -> str:
        tokens = []
        for t in proof_text.split():
            if re.fullmatch(r"sent\d+", t):
                i = int(t[4:])
                tokens.append(f"sent{permutation[i-1]+1}")
            else:
                tokens.append(t)
        renamed_proof = " ".join(tokens)
        return renamed_proof

    def shuffle_proof(self, proof_text, inv_permutation) -> str:
        tokens = []
        for t in proof_text.split():
            if re.fullmatch(r"sent\d+", t):
                i = int(t[4:])
                tokens.append(f"sent{inv_permutation[i-1]+1}")
            else:
                tokens.append(t)
        renamed_proof = " ".join(tokens)
        return renamed_proof

    def next_int(self) -> str:
        """
        Return the next intermediate conclusion identifier unused in the proof.
        """
        existing_ints = {step.conclusion_ident for step in self.proof_steps}
        for n in itertools.count(1):
            ident = f"int{n}"
            if ident not in existing_ints:
                break
        return ident

    def is_complete(self) -> bool:
        return self.proof_text.endswith("-> hypothesis")

    def find(self, premises) -> int:
        cls_ids = []
        for premise in premises:
            if re.fullmatch(r"sent\d+", premise):
                i = int(premise[4:])
                cls_ids.append(int(i))
        return cls_ids

    def random_select(self, premises) -> str:
        ind = random.choice(np.arange(len(premises)))
        check = False
        for premise in list(self.context.keys()):
            if premise not in set(premises):
                check = True
                break
        if not check:
            return None, None

        while True:
            random_premise = random.choice(list(self.context.keys()))
            if random_premise not in set(premises):
                new_premises = deepcopy(premises)
                new_premises[ind] = random_premise
                break
        return random_premise, new_premises

    def block_context(self, partial_proof, stepwise_goal) -> str:
        tokens = set()
        for t in partial_proof.split():
            if re.fullmatch(r"sent\d+", t):
                tokens.add(t)
        for t in stepwise_goal.split():
            if re.fullmatch(r"sent\d+", t):
                tokens.add(t)
        new_context = OrderedDict()
        for key in self.context.keys():
            if key in tokens:
                new_context[key] = self.context[key]
            else:
                if random.random() < 0.7:
                    new_context[key] = self.context[key]
        return normalize(" ".join(f"{k}: {v}" for k, v in new_context.items()))

