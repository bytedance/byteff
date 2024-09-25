# -----  ByteFF: ByteDance Force Field -----
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import decimal
import itertools
import logging
import os
from copy import deepcopy
from enum import IntEnum
from typing import Iterable, Literal, Tuple, Union
from uuid import uuid4

import numpy as np

# pylint: disable=W0201

logger = logging.getLogger(__name__)


class TopoDuplicateNameAtomTypeException(Exception):

    def __init__(self, name: str) -> None:
        msg = f"Duplicate name atomtype {name}"
        super().__init__(msg)


class TopoDuplicateNameMoleculeTypeException(Exception):

    def __init__(self, name: str) -> None:
        msg = f"Duplicate name moleculetype {name}"
        super().__init__(msg)


class Factory:
    _instances = {}

    def __new__(cls, uuid=None):
        k = uuid4() if uuid is None else uuid
        if (cls, k) not in cls._instances:
            instance = super().__new__(cls)
            instance.uuid = k
            instance._initialized = False
            cls._instances[(cls, k)] = instance
        return cls._instances[(cls, k)]

    @classmethod
    def remove(cls, uuid):
        if (cls, uuid) in cls._instances:
            del cls._instances[(cls, uuid)]


###########
# records #
###########


class RecordText:

    def _check(self):
        pass

    def _init(self, **kwargs):
        """
        line: None or str. Passed in as "text=line".
            line will be stripped, then split by semicolon into text and comment.
            text will be split by whitespace into _fields.
        verbose: None or bool. Passed in as "verbose=verbose".
            In __str__, comment will be appended to the output only if True.
            If set to False, comment will never be print, unless verbose is set to True again.
        comment: None or str. Passed in as "comment=comment".
            If not None, this will overwrite the comment found in `line'.
        """

        self.line: str = kwargs.get("text", None)
        self.verbose: bool = kwargs.get("verbose", None)

        self.comment: str = None
        self.text: str = None
        self._fields: list[str] = []

        if isinstance(self.line, str):
            self.line = self.line.rstrip().lstrip()
            self._split(line=self.line)
        self.comment = kwargs.get("comment", self.comment)

    def _split(self, line: str):
        n = len(line)
        i = 0
        while i < n:
            if line[i] == ";":
                break
            i = i + 1
        end = i - 1
        while end > -1:
            if line[end] != " ":
                break
            end = end - 1

        self.comment = line[end + 1:]
        self.text = line[:end + 1]
        self._fields = self.text.split()

    #########
    # print #
    #########

    def _str_impl(self):
        return self.text

    def __str__(self) -> str:
        line = self._str_impl()
        if self.verbose and self.comment is not None:
            line = line + self.comment
        return line

    ############
    # __init__ #
    ############

    def __init__(self, **kwargs):
        self._init(**kwargs)
        self._check()

    @classmethod
    def from_text(cls, text: str, **kwargs):
        r = super().__new__(cls)
        r._init(text=text, **kwargs)
        r._parse_line()
        r._check()
        return r

    def _parse_line(self):
        pass


class RecordInclude(RecordText):

    def _init(self, dirname: str, include: str = None, **kwargs):
        super()._init(**kwargs)

        self.dirname = dirname
        self.include = include
        self.abspath = None

    def _parse_line(self):
        fields = self._fields

        inc = fields[1]
        inc = inc[1:-1]
        self.include = inc
        if os.path.isabs(inc):
            self.abspath = self.include
        else:
            path = f"{self.dirname}/{inc}"
            path = os.path.abspath(path)
            self.abspath = path

    def _str_impl(self):
        return f"#include \"{self.include}\""


class RecordSection(RecordText):

    allowed_sections = (
        "defaults",
        "atomtypes",
        ##
        "moleculetype",
        "atoms",
        # "bondtypes",
        "bonds",
        # "angletypes",
        "angles",
        # "dihedraltypes",
        "dihedrals",
        "settles",
        "exclusions",
        # "pairtypes",
        "pairs",
        # "constrainttypes",
        # "constraints",
        # "nonbond_params",
        "virtual_sites1",
        "virtual_sites2",
        "virtual_sites3",
        "virtual_sites4",
        # "virtual_sitesn",
        # "*restraints*",
        ##
        "system",
        "molecules",
    )

    def _check(self):
        if self.section is None:
            return
        if self.section not in self.allowed_sections:
            if self.allow_unknown:
                logger.warning(f"Ignoring section [ {self.section} ].")
            else:
                raise NotImplementedError(f"Section [ {self.section} ] is not supported.")

    def _init(self, section: str = None, allow_unknown: bool = False, **kwargs):
        super()._init(**kwargs)
        self.section = section
        self.allow_unknown = allow_unknown

    def _parse_line(self):
        fields = self._fields

        self.section = fields[1]

    def _str_impl(self):
        return f"[ {self.section} ]"


############
# defaults #
############


class NonbondedFunctionEnum(IntEnum):
    """
    nbfunc
    """
    LENNARD_JONES = 1
    # BUCKINGHAM = 2


class LJCombinationRuleEnum(IntEnum):
    """
    LJ = C6/r**6 - C12/r**12
       = 4 eps [(sig/r)**6 - (sig/r)**12]
    C6, C12 = 4*eps*sig**6, 4*eps*sig**12
    sig, eps = (C12/C6)**(1/6), C6**2/(4*C12)
    comb-rule
        vdw-def         comb-rule
    1   V,W = C6,C12    C6,C12 = geometric,geometric
    2   V,W = sig,eps   sig,eps = arithmetic,geometric
    3   V,W = sig,eps   C6,C12 = geometric,geometric

    Buckingham = A exp(-Br) - C/r**6
    comb-rule
    A,B,C = geometric,harmonic,geometric
    """
    # C6_C12_GEO_GEO = 1
    SIGMA_EPSILON = 2
    # SIGMA_EPSILON_C6_C12_GEO_GEO = 3


class RecordDefaults(RecordText):

    def _check(self):
        assert self.gen_pairs in ("yes", "no")
        assert 0 < self.fudge_lj <= 1
        assert 0 < self.fudge_qq <= 1

    def _init(self,
              nbfunc: NonbondedFunctionEnum = NonbondedFunctionEnum.LENNARD_JONES,
              comb_rule: LJCombinationRuleEnum = LJCombinationRuleEnum.SIGMA_EPSILON,
              gen_pairs: str = "yes",
              fudge_lj: float = 0.5,
              fudge_qq: float = 1.0 / 1.2,
              **kwargs):
        super()._init(**kwargs)

        self.nbfunc, self.comb_rule = nbfunc, comb_rule
        self.gen_pairs = gen_pairs
        self.fudge_lj, self.fudge_qq = fudge_lj, fudge_qq  # vdw 14 and charge 14 scales
        if isinstance(self.nbfunc, int):
            self.nbfunc = NonbondedFunctionEnum(self.nbfunc)
        if isinstance(self.comb_rule, int):
            self.comb_rule = LJCombinationRuleEnum(self.comb_rule)
        if isinstance(self.gen_pairs, str):
            self.gen_pairs = str.lower(self.gen_pairs)

    def _parse_line(self):
        fields = self._fields

        nfields = len(fields)
        self.nbfunc = NonbondedFunctionEnum(int(fields[0]))
        self.comb_rule = LJCombinationRuleEnum(int(fields[1]))
        self.gen_pairs = str.lower(fields[2])
        if nfields > 3:
            self.fudge_lj = float(fields[3])
        if nfields > 4:
            self.fudge_qq = float(fields[4])

    def annotate(self) -> str:
        return ";     nbfunc   comb-rule   gen-pairs     fudgeLJ     fudgeQQ"

    def _str_impl(self):
        return f"{self.nbfunc:>12}{self.comb_rule:>12}{self.gen_pairs:>12}{self.fudge_lj:>12.8f}{self.fudge_qq:>12.8f}"


class TopoDefaults(Factory):

    def __init__(self, uuid=None):
        if not self._initialized:
            self._initialized = True
            self._added = False
            self.item: RecordDefaults = RecordDefaults()

    def __str__(self) -> str:
        section = RecordSection(section="defaults")
        lines = []
        lines.append("")
        lines.append(str(section))
        lines.append(self.item.annotate())
        lines.append(str(self.item))
        return "\n".join(lines)

    def add_record(self, record: RecordDefaults) -> int:
        assert isinstance(record, RecordDefaults)
        if not self._added:
            self._added = True
            self.item = deepcopy(record)
            return 0
        else:
            raise RuntimeError("A [ defaults ] section has been parsed.")

    @property
    def nbfunc(self):
        return self.item.nbfunc

    @property
    def comb_rule(self):
        return self.item.comb_rule

    @property
    def gen_pairs(self):
        return self.item.gen_pairs

    @property
    def fudge_lj(self):
        return self.item.fudge_lj

    @property
    def fudge_qq(self):
        return self.item.fudge_qq


#############
# atomtypes #
#############


class RecordAtomType(RecordText):
    """
    type name; bonded type (optional); atomic number (optional); m; q; particle type; V; W

    alias (if comb_rule == SIGMA_EPSILON)
    V: sigma
    W: epsilon
    """

    def _check(self):
        if isinstance(self.particle_type, str):
            assert self.particle_type in ("A", "V")

    def _init(self,
              name: str = None,
              at_num: int = 0,
              mass: float = 0.0,
              charge: float = 0.0,
              ptype: str = "A",
              V: float = 1.0,
              W: float = 0.0,
              **kwargs):
        super()._init(**kwargs)

        self.name = name
        # self.bonded_type = None
        self.at_num = at_num
        self.mass, self.charge = mass, charge
        self.particle_type = ptype
        self.V, self.W = V, W

    def _parse_line(self):
        fields = self._fields

        nfields = len(fields)
        self.name = fields[0]
        b = 1
        if nfields == 8:
            self.bonded_type, self.at_num = fields[1], int(fields[2])
            b = 3
        elif nfields == 7:
            if fields[1].isdigit():
                self.at_num = int(fields[1])
            else:
                self.bonded_type = fields[1]
                delattr(self, "at_num")
            b = 2
        self.mass, self.charge = float(fields[b + 0]), float(fields[b + 1])
        self.particle_type = fields[b + 2]
        self.V, self.W = float(fields[b + 3]), float(fields[b + 4])

    def annotate(self, comb_rule: LJCombinationRuleEnum) -> str:
        line = ";name           bonded.type        at.num        mass        charge   ptype"
        if comb_rule == LJCombinationRuleEnum.SIGMA_EPSILON:
            return line + "           sigma         epsilon"
        else:
            return line + "           V(c6)          W(c12)"

    def _str_impl(self):
        line = f" {self.name:<15}"
        if hasattr(self, "bonded_type") and hasattr(self, "at_num"):
            line = line + f"{self.bonded_type:>15}{self.at_num:>10}"
        elif hasattr(self, "at_num"):
            line = line + f"               {self.at_num:>10}"
        elif hasattr(self, "bonded_type"):
            line = line + f"{self.bonded_type:>15}          "
        line = line + f"{self.mass:>12.4f}{self.charge:>14.5f}{self.particle_type:>8}{self.V:>16.5e}{self.W:>16.5e}"
        return line

    @property
    def sigma(self):
        return self.V

    @sigma.setter
    def sigma(self, value):
        self.V = value

    @property
    def epsilon(self):
        return self.W

    @epsilon.setter
    def epsilon(self, value):
        self.W = value


class TopoAtomTypes(Factory):

    def __init__(self, uuid=None):
        if not self._initialized:
            self._initialized = True
            self.atomtypes: list[RecordAtomType] = []
            self.type_to_index: dict[str, int] = {}

    @property
    def atom_types(self):
        return self.atomtypes

    def copy_from(self, ta):
        if self.uuid == ta.uuid:
            return
        self.atomtypes = [a for a in ta.atomtypes]
        self.type_to_index = deepcopy(ta.type_to_index)

    def add_record(self, record: RecordAtomType) -> int:
        """
        Return 0-based index where the new atomtype can be found.
        """
        assert isinstance(record, RecordAtomType)
        if record.name in self.type_to_index.keys():
            raise TopoDuplicateNameAtomTypeException(name=record.name)
        idx = len(self.atomtypes)
        self.atomtypes.append(record)
        self.type_to_index[record.name] = idx
        if record.verbose is None:
            self.atomtypes[idx].verbose = True
        return idx

    def rename_atomtype(self, oldname: str, newname: str) -> int:
        """
        Return 0-based index where the atom type name is renamed in-place.
        """
        if newname in self.type_to_index.keys():
            raise TopoDuplicateNameAtomTypeException(newname)
        idx = self.type_to_index[oldname]
        self.atomtypes[idx].name = newname
        self.type_to_index.pop(oldname)
        self.type_to_index[newname] = idx
        return idx

    def str_slice(self, select) -> str:
        lines = []
        if isinstance(select, slice):
            atps = self.atomtypes[select]
        else:
            atps = [self.atomtypes[i] for i in select]
        if len(atps):
            td = TopoDefaults(self.uuid)
            section = RecordSection(section="atomtypes")
            lines.append("")
            lines.append(str(section))
            lines.append(self.atomtypes[0].annotate(td.comb_rule))
            for a in atps:
                lines.append(str(a))
        return "\n".join(lines)

    def __str__(self) -> str:
        select = slice(0, None)
        return self.str_slice(select)


##########################################
# atoms, moleculetype, molecules, system #
##########################################


class RecordAtom(RecordText):

    def _check(self):
        # limited by gro file specs
        if isinstance(self.residue, str):
            if len(self.residue) > 5:
                logger.warning(f"Residue longer than 5 characters: {self.residue}")
        if isinstance(self.atom, str):
            if len(self.atom) > 5:
                logger.warning(f"Atom name longer than 5 characters: {self.atom}")

    def _init(self,
              nr: int = None,
              atype: str = None,
              resnr: int = None,
              residue: str = None,
              atom: str = None,
              cgnr: int = None,
              charge: float = None,
              mass: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.nr, self.atype = nr, atype  # number, atomtype
        self.resnr, self.residue = resnr, residue  # residue number, residue name
        self.atom, self.cgnr = atom, cgnr  # atom name, charge group number
        if charge is not None:
            self.charge = charge
        if mass is not None:
            self.mass = mass

    def _parse_line(self):
        fields = self._fields

        nfields = len(fields)
        self.nr, self.atype = int(fields[0]), fields[1]
        self.resnr, self.residue = int(fields[2]), fields[3]
        self.atom, self.cgnr = fields[4], int(fields[5])
        if nfields > 6:
            self.charge = float(fields[6])
        if nfields > 7:
            self.mass = float(fields[7])

    def annotate(self) -> str:
        line = ";       nr            type      resi   res      atom      cgnr"
        if hasattr(self, "charge"):
            line = line + "      charge"
        if hasattr(self, "mass"):
            line = line + "          mass"
        return line

    def _str_impl(self):
        line = f"{self.nr:>10}{self.atype:>16}{self.resnr:>10}{self.residue:>6}{self.atom:>10}{self.cgnr:>10}"
        if hasattr(self, "charge"):
            line = line + f"{self.charge:>12.5f}"
        if hasattr(self, "mass"):
            line = line + f"{self.mass:>14.4f}"
        return line


class RecordMoleculeType(RecordText):

    def _init(self, name: str = None, nrexcl: int = None, **kwargs):
        super()._init(**kwargs)

        self.name, self.nrexcl = name, nrexcl

    def _parse_line(self):
        fields = self._fields

        self.name = fields[0]
        self.nrexcl = int(fields[1])

    def annotate(self) -> str:
        return ";name             nrexcl"

    def _str_impl(self):
        return f" {self.name:<15}{self.nrexcl:>8}"


class RecordMolecule(RecordText):

    def _init(self, name: str = None, nr: int = None, **kwargs):
        super()._init(**kwargs)

        self.name, self.nr = name, nr  # molecule name, number of molecules

    def _parse_line(self):
        fields = self._fields

        self.name, self.nr = fields[0], int(fields[1])

    def annotate(self) -> str:
        return ";molecule            nmols"

    def _str_impl(self):
        return f" {self.name:<15}{self.nr:>10}"


class RecordSystem(RecordText):

    def _init(self, title: str = None, **kwargs):
        super()._init(**kwargs)

        self.title = title

    def _parse_line(self):
        self.title = self.text

    def _str_impl(self):
        return self.title


############################
# bonds, angles, dihedrals #
# settles                  #
# exclusions, pairs        #
############################


class BondTypeEnum(IntEnum):
    BOND = 1
    # G96 = 2
    # MORSE = 3
    # CUBIC = 4
    # CONNECTION = 5
    # HARMONIC_POTENTIAL = 6
    # FENE = 7
    # TABULATED_EXCL = 8
    # TABULATED_WO_EXCL = 9
    # RESTRAINT = 10


class RecordBond(RecordText):

    def _init(self,
              ai: int = None,
              aj: int = None,
              funct: BondTypeEnum = None,
              c0: float = None,
              c1: float = None,
              **kwargs):
        super()._init(**kwargs)

        if isinstance(ai, int):
            self.ai = ai
        if isinstance(aj, int):
            self.aj = aj
        self.funct = funct
        self.c0, self.c1 = c0, c1
        if isinstance(self.funct, int):
            self.funct = BondTypeEnum(self.funct)

    def raise_not_impl(self):
        raise NotImplementedError(f"Bond funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields

        if fields[0].isdigit() and fields[1].isdigit():
            self.ai, self.aj = int(fields[0]), int(fields[1])
        self.funct = BondTypeEnum(int(fields[2]))
        if self.funct == BondTypeEnum.BOND:
            self.c0, self.c1 = float(fields[3]), float(fields[4])
        else:
            self.raise_not_impl()

    def cmp_key(self):
        return (self.ai, self.aj)

    def annotate(self) -> str:
        if self.funct == BondTypeEnum.BOND:
            return ";       ai        aj funct               r               k"
        else:
            self.raise_not_impl()

    def _str_impl(self):
        if self.funct == BondTypeEnum.BOND:
            line = f"{self.ai:>10}{self.aj:>10}"
            line = line + f"{self.funct:>6}{self.c0:>16.6f}{self.c1:>16.6f}"
            return line
        else:
            self.raise_not_impl()

    @property
    def b0(self):
        assert self.funct in (BondTypeEnum.BOND,)
        return self.c0

    @b0.setter
    def b0(self, value):
        assert self.funct in (BondTypeEnum.BOND,)
        self.c0 = value

    @property
    def kb(self):
        assert self.funct in (BondTypeEnum.BOND,)
        return self.c1

    @kb.setter
    def kb(self, value):
        assert self.funct in (BondTypeEnum.BOND,)
        self.c1 = value


class PairTypeEnum(IntEnum):
    EXTRA_LJ = 1
    EXTRA_COULOMB_LJ = 2


class RecordPair(RecordText):

    def _init(self,
              ai: int = None,
              aj: int = None,
              funct: PairTypeEnum = None,
              V: float = None,
              W: float = None,
              fudge_qq: float = None,
              qi: float = None,
              qj: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.ai, self.aj = ai, aj
        self.funct = funct
        if V is not None:
            self.V = V
        if W is not None:
            self.W = W
        if fudge_qq is not None:
            self.fudge_qq = fudge_qq
        if qi is not None:
            self.qi = qi
        if qj is not None:
            self.qj = qj
        if isinstance(self.funct, int):
            self.funct = PairTypeEnum(self.funct)

    def get_params(self, td: TopoDefaults, vws: list[Tuple[float, float]],
                   charges: list[float]) -> Tuple[float, float, float, float, float]:
        fudge_lj = td.fudge_lj
        fudge_qq, qi, qj = td.fudge_qq, 0.0, 0.0
        v, w = 1.0, 0.0

        ai0, aj0 = self.ai - 1, self.aj - 1

        if hasattr(self, "V") and hasattr(self, "W"):
            v, w = self.V, self.W
        else:
            assert td.nbfunc == NonbondedFunctionEnum.LENNARD_JONES
            assert td.comb_rule == LJCombinationRuleEnum.SIGMA_EPSILON

            vi, wi = vws[ai0]
            vj, wj = vws[aj0]
            v = (vi + vj) / 2
            w = (wi * wj)**0.5
            w *= fudge_lj

        if hasattr(self, "fudge_qq") and hasattr(self, "qi") and hasattr(self, "qj"):
            fudge_qq, qi, qj = self.fudge_qq, self.qi, self.qj
        else:
            qi, qj = charges[ai0], charges[aj0]

        return (v, w, fudge_qq, qi, qj)

    def _parse_line(self):
        fields = self._fields

        nfields = len(fields)
        self.ai, self.aj = int(fields[0]), int(fields[1])
        self.funct = PairTypeEnum(int(fields[2]))
        if self.funct == PairTypeEnum.EXTRA_LJ:
            if nfields > 3:
                self.V, self.W = float(fields[3]), float(fields[4])
        elif self.funct == PairTypeEnum.EXTRA_COULOMB_LJ:
            self.fudge_qq, self.qi, self.qj = float(fields[3]), float(fields[4]), float(fields[5])
            self.V, self.W = float(fields[6]), float(fields[7])

    def cmp_key(self):
        return (self.ai, self.aj)

    def annotate(self) -> str:
        if self.funct == PairTypeEnum.EXTRA_LJ:
            if hasattr(self, "V") and hasattr(self, "W"):
                return ";       ai        aj funct               V               W"
            else:
                return ";       ai        aj funct"
        elif self.funct == PairTypeEnum.EXTRA_COULOMB_LJ:
            return ";       ai        aj funct         fudgeQQ              qi              qj               V               W"

    def _str_impl(self):
        line = f"{self.ai:>10}{self.aj:>10}{self.funct:>6}"
        if self.funct == PairTypeEnum.EXTRA_LJ:
            if hasattr(self, "V") and hasattr(self, "W"):
                return line + f"{self.V:>16.6f}{self.W:>16.6f}"
            else:
                return line
        elif self.funct == PairTypeEnum.EXTRA_COULOMB_LJ:
            return line + f"{self.fudge_qq:>16.6f}{self.qi:>16.6f}{self.qj:>16.6f}{self.V:>16.6f}{self.W:>16.6f}"
        else:
            self.raise_not_impl()


class AngleTypeEnum(IntEnum):
    ANGLE = 1
    # G96 = 2
    # CROSS_BOND_BOND = 3
    # CROSS_BOND_ANGLE = 4
    # UREY_BRADLEY = 5
    # QUARTIC = 6
    # TABULATED = 8
    # LINEAR = 9
    # RESTRICTED_BENDING = 10


class RecordAngle(RecordText):

    def _init(self,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              funct: AngleTypeEnum = None,
              c0: float = None,
              c1: float = None,
              **kwargs):
        super()._init(**kwargs)

        if isinstance(ai, int):
            self.ai = ai
        if isinstance(aj, int):
            self.aj = aj
        if isinstance(ak, int):
            self.ak = ak
        self.funct = funct
        self.c0, self.c1 = c0, c1
        if isinstance(self.funct, int):
            self.funct = AngleTypeEnum(self.funct)

    def raise_not_impl(self):
        raise NotImplementedError(f"Angle funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields

        if fields[0].isdigit() and fields[1].isdigit() and fields[2].isdigit():
            self.ai, self.aj, self.ak = int(fields[0]), int(fields[1]), int(fields[2])
        self.funct = AngleTypeEnum(int(fields[3]))
        if self.funct == AngleTypeEnum.ANGLE:
            self.c0, self.c1 = float(fields[4]), float(fields[5])
        else:
            self.raise_not_impl()

    def cmp_key(self):
        return (self.ai, self.aj, self.ak)

    def annotate(self) -> str:
        if self.funct == AngleTypeEnum.ANGLE:
            return ";       ai        aj        ak funct           theta             cth"
        else:
            self.raise_not_impl()

    def _str_impl(self):
        if self.funct == AngleTypeEnum.ANGLE:
            line = f"{self.ai:>10}{self.aj:>10}{self.ak:>10}"
            line = line + f"{self.funct:>6}{self.c0:>16.6f}{self.c1:>16.6f}"
            return line
        else:
            self.raise_not_impl()

    @property
    def theta(self):
        assert self.funct in (AngleTypeEnum.ANGLE,)
        return self.c0

    @theta.setter
    def theta(self, value):
        assert self.funct in (AngleTypeEnum.ANGLE,)
        self.c0 = value

    @property
    def k(self):
        assert self.funct in (AngleTypeEnum.ANGLE,)
        return self.c1

    @k.setter
    def k(self, value):
        assert self.funct in (AngleTypeEnum.ANGLE,)
        self.c1 = value


class DihedralTypeEnum(IntEnum):
    PROPER = 1
    # IMPROPER = 2
    RYCKAERT_BELLEMANS = 3
    PERIODIC_IMPROPER = 4
    # FOURIER = 5
    # TABULATED = 8
    MULTIPLE_PROPER = 9
    # RESTRICTED = 10
    # COMBINED_BENDING_TORSION = 11


class RecordDihedral(RecordText):

    def _check(self):
        if self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                          DihedralTypeEnum.MULTIPLE_PROPER):
            assert isinstance(self.c2, int)

    def _init(self,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              al: int = None,
              funct: DihedralTypeEnum = None,
              c0: float = None,
              c1: float = None,
              c2: Union[int, float] = None,
              c3: float = None,
              c4: float = None,
              c5: float = None,
              **kwargs):
        super()._init(**kwargs)

        if isinstance(ai, int):
            self.ai = ai
        if isinstance(aj, int):
            self.aj = aj
        if isinstance(ak, int):
            self.ak = ak
        if isinstance(al, int):
            self.al = al
        self.funct = funct
        self.c0, self.c1, self.c2 = c0, c1, c2
        self.c3, self.c4, self.c5 = c3, c4, c5
        if isinstance(self.funct, int):
            self.funct = DihedralTypeEnum(self.funct)

    def raise_not_impl(self):
        raise NotImplementedError(f"Dihedral funct {self.funct} ({self.funct.name}) is not implemented.")

    def _parse_line(self):
        fields = self._fields

        if fields[0].isdigit() and fields[1].isdigit() and fields[2].isdigit() and fields[3].isdigit():
            self.ai, self.aj, self.ak, self.al = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
        self.funct = DihedralTypeEnum(int(fields[4]))
        if self.funct == DihedralTypeEnum.RYCKAERT_BELLEMANS:
            self.c0 = float(fields[5])
            self.c1 = float(fields[6])
            self.c2 = float(fields[7])
            self.c3 = float(fields[8])
            self.c4 = float(fields[9])
            self.c5 = float(fields[10])
        elif self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                            DihedralTypeEnum.MULTIPLE_PROPER):
            self.c0 = float(fields[5])
            self.c1 = float(fields[6])
            self.c2 = int(fields[7])
        else:
            self.raise_not_impl()

    def cmp_key(self):
        return (self.ai, self.aj, self.ak, self.al)

    def annotate(self) -> str:
        line = ";       ai        aj        ak        al funct"
        if self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                          DihedralTypeEnum.MULTIPLE_PROPER):
            line = line + "           phase               k        pn"
            return line
        elif self.funct in (DihedralTypeEnum.RYCKAERT_BELLEMANS,):
            line = line + "               c"
        else:
            self.raise_not_impl()

    def _str_impl(self):
        line = f"{self.ai:>10}{self.aj:>10}{self.ak:>10}{self.al:>10}"
        line = line + f"{self.funct:>6}"
        if self.funct == DihedralTypeEnum.RYCKAERT_BELLEMANS:
            return line + f"{self.c0:>16.6f}{self.c1:>16.6f}{self.c2:>16.6f}{self.c3:>16.6f}{self.c4:>16.6f}{self.c5:>16.6f}"
        elif self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                            DihedralTypeEnum.MULTIPLE_PROPER):
            return line + f"{self.c0:>16.6f}{self.c1:>16.6f}{self.c2:>10}"
        else:
            self.raise_not_impl()

    @property
    def phi(self):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER)
        return self.c0

    @phi.setter
    def phi(self, value):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER)
        self.c0 = value

    @property
    def k(self):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER)
        return self.c1

    @k.setter
    def k(self, value):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER)
        self.c1 = value

    @property
    def multiplicity(self):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER)
        return self.c2

    @multiplicity.setter
    def multiplicity(self, value):
        assert self.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.PERIODIC_IMPROPER,
                              DihedralTypeEnum.MULTIPLE_PROPER)
        self.c2 = value


class RecordSettle(RecordText):

    def _check(self):
        assert self.funct == 1

    def _init(self, ao: int = None, funct: int = 1, doh: float = None, dhh: float = None, **kwargs):
        super()._init(**kwargs)

        self.idx = ao
        self.funct = funct
        self.doh, self.dhh = doh, dhh

    def _parse_line(self):
        fields = self._fields

        self.idx = int(fields[0])
        self.funct = int(fields[1])
        self.doh = float(fields[2])
        self.dhh = float(fields[3])

    def annotate(self) -> str:
        return ";             OW funct             dOH             dHH"

    def _str_impl(self):
        return f"{self.idx:>16}{self.funct:>6}{self.doh:>16.6f}{self.dhh:>16.6f}"


class RecordExclusion(RecordText):

    def _check(self):
        assert self.ai not in self.aj_list

    def _init(self, ai: int = None, aj_list: list[int] = None, **kwargs):
        super()._init(**kwargs)
        if aj_list is None:
            aj_list = []
        self.ai, self.aj_list = ai, [aj for aj in aj_list]
        self.aj_list.sort()

    def _parse_line(self):
        fields = self._fields

        self.ai = int(fields[0])
        self.aj_list = [int(aj) for aj in fields[1:]]
        self.aj_list.sort()

    def cmp_key(self):
        return self.ai

    def _str_impl(self):
        return " ".join([f"{j:>8}" for j in ([self.ai] + self.aj_list)])


#################
# virtual sites #
#################


class VirtualSite2Enum(IntEnum):
    _2 = 1
    _2FD = 2


class VirtualSite3Enum(IntEnum):
    _3 = 1
    _3FD = 2
    _3FAD = 3
    _3OUT = 4


class VirtualSite4Enum(IntEnum):
    _4FDN = 2


class RecordVirtualSite1(RecordText):

    def _check(self):
        assert self.funct == 1

    def _init(self, av: int = None, ai: int = None, funct: int = 1, **kwargs):
        super()._init(**kwargs)

        self.av, self.ai = av, ai
        self.funct = funct

    def _parse_line(self):
        fields = self._fields

        self.av, self.ai = int(fields[0]), int(fields[1])
        self.funct = int(fields[2])

    def annotate(self) -> str:
        return ";     site         i funct"

    def _str_impl(self):
        return f"{self.av:>10}{self.ai:>10}{self.funct:>6}"


class RecordVirtualSite2(RecordText):

    def _init(self,
              av: int = None,
              ai: int = None,
              aj: int = None,
              funct: VirtualSite2Enum = None,
              c: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.av, self.ai, self.aj = av, ai, aj
        self.funct = funct
        self.c = c
        if isinstance(self.funct, int):
            self.funct = VirtualSite2Enum(self.funct)

    def _parse_line(self):
        fields = self._fields

        self.av, self.ai, self.aj = int(fields[0]), int(fields[1]), int(fields[2])
        self.funct = VirtualSite2Enum(int(fields[3]))
        self.c = float(fields[4])

    def annotate(self) -> str:
        return ";     site        ij           funct          params"

    def _str_impl(self):
        return f"{self.av:>10}{self.ai:>10}{self.aj:>10}{self.funct:>6}{self.c:>16.6f}"


class RecordVirtualSite3(RecordText):

    def _init(self,
              av: int = None,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              funct: VirtualSite3Enum = None,
              c0: float = None,
              c1: float = None,
              c2: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.av, self.ai, self.aj, self.ak = av, ai, aj, ak
        self.funct = funct
        self.c0, self.c1, self.c2 = c0, c1, c2
        if isinstance(self.funct, int):
            self.funct = VirtualSite3Enum(self.funct)

    def _parse_line(self):
        fields = self._fields

        self.av, self.ai, self.aj, self.ak = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
        self.funct = VirtualSite3Enum(int(fields[4]))
        self.c0, self.c1 = float(fields[5]), float(fields[6])
        if self.funct == VirtualSite3Enum._3OUT:
            self.c2 = float(fields[7])

    def annotate(self) -> str:
        return ";     site       ijk                     funct          params"

    def _str_impl(self):
        line = f"{self.av:>10}{self.ai:>10}{self.aj:>10}{self.ak:>10}{self.funct:>6}{self.c0:>16.6f}{self.c1:>16.6f}"
        if self.funct == VirtualSite3Enum._3OUT:
            line = line + f"{self.c2:>16.6f}"
        return line


class RecordVirtualSite4(RecordText):

    def _init(self,
              av: int = None,
              ai: int = None,
              aj: int = None,
              ak: int = None,
              al: int = None,
              funct: VirtualSite4Enum = None,
              c0: float = None,
              c1: float = None,
              c2: float = None,
              **kwargs):
        super()._init(**kwargs)

        self.av, self.ai, self.aj, self.ak, self.al = av, ai, aj, ak, al
        self.funct = funct
        self.c0, self.c1, self.c2 = c0, c1, c2
        if isinstance(self.funct, int):
            self.funct = VirtualSite4Enum(self.funct)

    def _parse_line(self):
        fields = self._fields

        self.av = int(fields[0])
        self.ai, self.aj, self.ak, self.al = int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4])
        self.funct = VirtualSite4Enum(int(fields[5]))
        self.c0, self.c1, self.c2 = float(fields[6]), float(fields[7]), float(fields[8])

    def annotate(self) -> str:
        return ";     site      ijkl                               funct          params"

    def _str_impl(self):
        return f"{self.av:>10}{self.ai:>10}{self.aj:>10}{self.ak:>10}{self.al:>10}{self.funct:>6}{self.c0:>16.6f}{self.c1:>16.6f}{self.c2:>16.6f}"


###############
# all records #
###############


class Records:

    def __init__(self):
        self.all: list[RecordText] = []

    def __iadd__(self, other):
        assert isinstance(other, Records)
        self.all.extend(other.all)
        return self

    @classmethod
    def from_file(cls, path: str, incdir: str, allow_unknown: bool):
        path = os.path.abspath(path)
        basename = os.path.basename(path)
        dirname = os.path.dirname(path) if incdir is None else incdir

        from_text = (
            RecordDefaults.from_text,
            RecordAtomType.from_text,
            ##
            RecordMoleculeType.from_text,
            RecordAtom.from_text,
            # RecordBondType.from_text,
            RecordBond.from_text,
            # RecordAngleType.from_text,
            RecordAngle.from_text,
            # RecordDihedralType.from_text,
            RecordDihedral.from_text,
            RecordSettle.from_text,
            RecordExclusion.from_text,
            # RecordPairType.from_text,
            RecordPair.from_text,
            # RecordConstraintType.from_text,
            # RecordConstraint.from_text,
            # RecordNonbondParam.from_text,
            RecordVirtualSite1.from_text,
            RecordVirtualSite2.from_text,
            RecordVirtualSite3.from_text,
            RecordVirtualSite4.from_text,
            ##
            RecordSystem.from_text,
            RecordMolecule.from_text,
        )
        assert len(RecordSection.allowed_sections) == len(from_text)
        func_from_text = dict((k, v) for k, v in zip(RecordSection.allowed_sections, from_text))

        fs = cls()
        lineno = 0
        cur_section = None
        for raw in open(path):
            lineno += 1
            r0 = RecordText.from_text(text=raw)
            text = r0.text
            line = r0.line

            if text == "":
                fs.all.append(r0)
            elif text.startswith("#include"):
                r = RecordInclude.from_text(dirname=dirname, text=text)  # filter the comment after #include "file.path"
                fs.all.append(r)
                fs2 = cls.from_file(path=r.abspath, incdir=incdir, allow_unknown=allow_unknown)
                fs += fs2
            elif text.startswith("#"):
                if allow_unknown:
                    logger.warning(f"Ignoring {basename}:{lineno} {line}")
                else:
                    raise RuntimeError(f"Cannot parse {basename}:{lineno} {line}")
            elif text.startswith("[ ") and text.endswith(" ]"):
                r = RecordSection.from_text(allow_unknown=allow_unknown, text=line)
                fs.all.append(r)
                cur_section = r.section
            elif cur_section in ("atomtypes", "atoms"):
                r = func_from_text[cur_section](text=line)
                fs.all.append(r)
            elif cur_section in func_from_text.keys():
                r = func_from_text[cur_section](text=line)
                fs.all.append(r)
            else:
                if allow_unknown:
                    logger.warning(f"Ignoring {basename}:{lineno} {line}")
                else:
                    raise RuntimeError(f"Cannot parse {basename}:{lineno} {line}")

        return fs


################
# molecule itp #
################


class TopoMolecule:

    def __init__(self, system_uuid, *, round_on: Literal["", "r", "w", "rw"] = "w", allow_round_diff: float = None):
        self.moleculetype: RecordMoleculeType = None
        self.atoms: list[RecordAtom] = []
        self.bonds: list[RecordBond] = []
        self.angles: list[RecordAngle] = []
        self.dihedrals: list[RecordDihedral] = []
        self.settles: list[RecordSettle] = []
        self.exclusions: list[RecordExclusion] = []
        self.pairs: list[RecordPair] = []
        self.virtual_sites1: list[RecordVirtualSite1] = []
        self.virtual_sites2: list[RecordVirtualSite2] = []
        self.virtual_sites3: list[RecordVirtualSite3] = []
        self.virtual_sites4: list[RecordVirtualSite4] = []

        self.atom_to_type_index: list[int] = []
        self.system_uuid = system_uuid

        self.round_on: str = round_on
        self.allow_round_diff: float = allow_round_diff

    @property
    def molecule_type(self):
        return self.moleculetype

    @property
    def name(self) -> str:
        return self.moleculetype.name

    @property
    def nrexcl(self) -> int:
        return self.moleculetype.nrexcl

    @property
    def natoms(self) -> int:
        """
        Number of physical atoms and virtual sites.
        """
        return len(self.atoms)

    @property
    def nrealatoms(self) -> int:
        """
        Number of physical atoms.
        """
        ta = TopoAtomTypes(self.system_uuid)
        return len([idx for idx in self.atom_to_type_index if ta.atomtypes[idx].particle_type == "A"])

    @property
    def nvsites(self) -> int:
        """
        Number of virtual sites.
        """
        ta = TopoAtomTypes(self.system_uuid)
        return len([idx for idx in self.atom_to_type_index if ta.atomtypes[idx].particle_type == "V"])

    def __str__(self) -> str:
        if "w" in self.round_on:
            if self.allow_round_diff is None or self.allow_round_diff < 0:
                raise RuntimeError(f"Incorrect allow_round_diff (current: {self.allow_round_diff}).")
            elif self.allow_round_diff > 0:
                self.update_charge(partial_charges=None, symm=None, to_round=True)
            else:
                pass
        lines = []

        if self.moleculetype is not None:
            lines.append("")
            section = "moleculetype"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.moleculetype.annotate())
            lines.append(str(self.moleculetype))

        if self.natoms:
            lines.append("")
            section = "atoms"
            lines.append(str(RecordSection(section=section)))
            qtot = 0.0
            a = self.atoms[0]
            lines.append(a.annotate() + " ; qtot" if hasattr(a, "charge") else "")
            for a in self.atoms:
                if hasattr(a, "charge"):
                    qtot += a.charge
                    suffix = f" ; qtot {qtot:8.3f}"
                else:
                    suffix = ""
                lines.append(str(a) + suffix)

        if len(self.bonds):
            lines.append("")
            section = "bonds"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.bonds[0].annotate())
            for a in self.bonds:
                lines.append(str(a))

        if len(self.angles):
            lines.append("")
            section = "angles"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.angles[0].annotate())
            for a in self.angles:
                lines.append(str(a))

        if len(self.dihedrals):
            propers = []
            improps = []
            for a in self.dihedrals:
                if a.funct in (DihedralTypeEnum.PROPER, DihedralTypeEnum.MULTIPLE_PROPER):
                    propers.append(a)
                elif a.funct in (DihedralTypeEnum.PERIODIC_IMPROPER,):
                    improps.append(a)
                else:
                    a.raise_not_impl()

            if len(propers):
                lines.append("")
                section = "dihedrals"
                comment = " ; propers for gromacs 4.5 or higher, use funct 9"
                verbose = True
                lines.append(str(RecordSection(section=section, comment=comment, verbose=verbose)))
                lines.append(propers[0].annotate())
                for a in propers:
                    lines.append(str(a))

            if len(improps):
                lines.append("")
                section = "dihedrals"
                comment = " ; impropers"
                verbose = True
                lines.append(str(RecordSection(section=section, comment=comment, verbose=verbose)))
                lines.append(improps[0].annotate())
                for a in improps:
                    lines.append(str(a))

        if len(self.settles):
            lines.append("")
            section = "settles"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.settles[0].annotate())
            for a in self.settles:
                lines.append(str(a))

        if len(self.exclusions):
            lines.append("")
            section = "exclusions"
            lines.append(str(RecordSection(section=section)))
            for a in self.exclusions:
                lines.append(str(a))

        if len(self.pairs):
            lines.append("")
            section = "pairs"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.pairs[0].annotate())
            for a in self.pairs:
                lines.append(str(a))

        if len(self.virtual_sites1):
            lines.append("")
            section = "virtual_sites1"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.virtual_sites1[0].annotate())
            for a in self.virtual_sites1:
                lines.append(str(a))

        if len(self.virtual_sites2):
            lines.append("")
            section = "virtual_sites2"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.virtual_sites2[0].annotate())
            for a in self.virtual_sites2:
                lines.append(str(a))

        if len(self.virtual_sites3):
            lines.append("")
            section = "virtual_sites3"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.virtual_sites3[0].annotate())
            for a in self.virtual_sites3:
                lines.append(str(a))

        if len(self.virtual_sites4):
            lines.append("")
            section = "virtual_sites4"
            lines.append(str(RecordSection(section=section)))
            lines.append(self.virtual_sites4[0].annotate())
            for a in self.virtual_sites4:
                lines.append(str(a))

        return "\n".join(lines)

    def _sort_bonds(self):
        for a in self.bonds:
            if a.ai > a.aj:
                a.ai, a.aj = a.aj, a.ai
        self.bonds.sort(key=RecordBond.cmp_key)

    def _sort_angles(self):
        for a in self.angles:
            if a.ai > a.ak:
                a.ai, a.ak = a.ak, a.ai  # aj does not change
        self.angles.sort(key=RecordAngle.cmp_key)

    def _sort_dihedrals(self):
        for a in self.dihedrals:
            if a.funct == DihedralTypeEnum.PERIODIC_IMPROPER:
                continue  # do not modify improper
            if a.ai > a.al:
                a.ai, a.aj, a.ak, a.al = a.al, a.ak, a.aj, a.ai
        self.dihedrals.sort(key=RecordDihedral.cmp_key)

    def _sort_exclusions(self):
        self.exclusions.sort(key=RecordExclusion.cmp_key)

    def _sort_pairs(self):
        for a in self.pairs:
            if a.ai > a.aj:
                a.ai, a.aj = a.aj, a.ai
        self.pairs.sort(key=RecordPair.cmp_key)

    def _check_pair_exclusion(self):
        excl_set = set()
        for excl in self.exclusions:
            for aj in excl.aj_list:
                excl_set.add((excl.ai, aj))
        for pair in self.pairs:
            ai, aj = pair.ai, pair.aj
            if (ai, aj) in excl_set or (aj, ai) in excl_set:
                logger.warning(f"Excluded pair re-added by [ pairs ]: ({ai:>10},{aj:>10})")

    def add_record_atom(self, record: RecordAtom, atomtype: RecordAtomType = None) -> int:
        ta = TopoAtomTypes(self.system_uuid)
        if record.atype in ta.type_to_index.keys():
            self.atoms.append(record)
            self.atom_to_type_index.append(ta.type_to_index[record.atype])
        else:
            assert atomtype is not None and record.atype == atomtype.name
            logger.info(f"Adding new type {record.atype} to the [ atomtypes ] section.")
            idx = ta.add_record(atomtype)
            self.atoms.append(record)
            self.atom_to_type_index.append(idx)
        return self.natoms - 1

    def add_record_vsite(self, record: RecordAtom, vrecord) -> int:
        self.add_record_atom(record=record)
        if isinstance(vrecord, RecordVirtualSite1):
            self.virtual_sites1.append(vrecord)
        elif isinstance(vrecord, RecordVirtualSite2):
            self.virtual_sites2.append(vrecord)
        elif isinstance(vrecord, RecordVirtualSite3):
            self.virtual_sites3.append(vrecord)
        elif isinstance(vrecord, RecordVirtualSite4):
            self.virtual_sites4.append(vrecord)
        else:
            assert False
        return self.natoms - 1

    def determine_vsite_exclusions_and_pairs(self):
        # this code only works if gen-pairs is "yes"
        assert TopoDefaults(self.system_uuid).gen_pairs == "yes"

        nb1x = self.get_nb1x()
        b11, b12, b13, b14 = nb1x["11"], nb1x["12"], nb1x["13"], nb1x["14"]
        ptypes = self.get_ptypes()

        # exclusions
        ai_idx = dict((a.ai, idx) for idx, a in enumerate(self.exclusions))
        for a0, pt in enumerate(ptypes):
            if pt == "V":
                ai = a0 + 1
                a1_list = b11[a0] + b12[a0] + b13[a0] + b14[a0]
                a1_list.sort()

                if ai not in ai_idx.keys():
                    idx = len(self.exclusions)
                    self.exclusions.append(RecordExclusion(ai=ai))
                else:
                    idx = ai_idx[ai]

                for a1 in a1_list:
                    aj = a1 + 1
                    if aj != ai and aj not in self.exclusions[idx].aj_list:
                        self.exclusions[idx].aj_list.append(aj)

                self.exclusions[idx].aj_list.sort()

        # pairs (1-4)
        current_pairs = set()
        for a in self.pairs:
            current_pairs.add((a.ai, a.aj))
            current_pairs.add((a.aj, a.ai))

        for a0, a1_list in b14.items():
            ai = a0 + 1
            for a1 in a1_list:
                aj = a1 + 1
                if (ai, aj) not in current_pairs:
                    current_pairs.add((ai, aj))
                    current_pairs.add((aj, ai))
                    self.pairs.append(RecordPair(ai=ai, aj=aj, funct=PairTypeEnum.EXTRA_LJ))

        self._sort_pairs()

    def set_atomtype(self, atomidx: int, atype_or_typeidx: Tuple[str, int]):
        ta = TopoAtomTypes(self.system_uuid)
        if isinstance(atype_or_typeidx, int):
            self.atoms[atomidx].atype = ta.atomtypes[atype_or_typeidx].name
            self.atom_to_type_index[atomidx] = atype_or_typeidx
        elif isinstance(atype_or_typeidx, str):
            self.atoms[atomidx].atype = atype_or_typeidx
            self.atom_to_type_index[atomidx] = ta.type_to_index[atype_or_typeidx]
        else:
            assert False

    @staticmethod
    def average_by_symmetry(data, symm: dict[int, list[int]] = None):
        arr = np.copy(data)
        if symm is not None:
            for _, atoms in symm.items():
                where = np.full(len(data), False)
                where[atoms] = True
                mean = np.mean(data, where=where)
                for a in atoms:
                    arr[a] = mean
        return arr

    @staticmethod
    def round_list_sum_to_int(lst: list[float], symm: dict[int, list[int]] = None, maxdiff: float = 0.01):
        lst = TopoMolecule.average_by_symmetry(data=lst, symm=symm)

        # convert the list of floats to a list of Decimals rounded to 5 decimal places
        decimal_context = decimal.Context(prec=28, rounding=decimal.ROUND_HALF_DOWN)
        decimal_lst = [decimal_context.create_decimal(str(x)).quantize(decimal.Decimal("1.00000")) for x in lst]

        # apply the adjustment to the last number to ensure the sum is an integer
        decimal_sum = sum(decimal_lst)
        gap = (round(decimal_sum) - decimal_sum)
        if abs(gap) > abs(maxdiff):
            msg = f"Not Rounded when List Sum to Integer abs(gap) > abs(maxdiff) {abs(gap):6.3f} >{abs(maxdiff):6.3f}."
            if abs(maxdiff) > 0:
                logger.warning(msg)
            return [x for x in lst]

        if decimal_sum % 1 != 0:
            decimal_lst[-1] += gap

        float_lst = [float(x) for x in decimal_lst]
        before = round(sum(lst))
        after = round(sum(float_lst))
        assert before == after, f"Rounding error before {before} after {after}"
        return float_lst

    def get_ivatoms(self) -> dict[int, list[int]]:
        """
        0-based iatom and associated vatoms.
        """
        ai_avs = {}
        ptypes = self.get_ptypes()
        for i, pt in enumerate(ptypes):
            if pt == "A":
                ai_avs[i] = {i}
        for vs in (self.virtual_sites1, self.virtual_sites2, self.virtual_sites3, self.virtual_sites4):
            for a in vs:
                av, ai = a.av - 1, a.ai - 1
                ai_avs[ai].add(av)
        return dict((ai, sorted(list(avs))) for ai, avs in ai_avs.items())

    def get_doppelgangers(self, symm: dict[int, list[int]]) -> dict[int, int]:
        """
        doppelgangers[jatom] = iatom, or jatom if unique
        i/jatom are 0-based.
        """
        doppelgangers = {i: i for i in range(self.natoms)}
        for iatom, eq_list in symm.items():
            for jatom in eq_list:
                doppelgangers[jatom] = iatom
        return doppelgangers

    def get_nb1x(self) -> dict[str, dict[int, list[int]]]:
        """
        0-based 1-x connected neighboring particles.
        """
        nb1x = {}

        natoms = self.natoms
        nrexcl = self.nrexcl
        boo = {i: {i} for i in range(natoms)}

        b11 = {i: [i] for i in range(natoms)}
        if nrexcl >= 0:
            ai_avs = self.get_ivatoms()
            for _ai, avs in ai_avs.items():
                if len(avs) > 1:
                    avs2 = list(avs)
                    for ai, aj in itertools.combinations(avs2, 2):
                        if ai not in boo[aj]:
                            b11[aj].append(ai)
                            boo[aj].add(ai)
                        if aj not in boo[ai]:
                            b11[ai].append(aj)
                            boo[ai].add(aj)

            for _k, v in b11.items():
                v.sort()
        nb1x["11"] = b11

        b12 = {i: [] for i in range(natoms)}
        if nrexcl >= 1:
            for a in self.bonds:
                a0, a1 = a.ai - 1, a.aj - 1
                for p in itertools.product(b11[a0], b11[a1]):
                    ai, aj = p[0], p[1]
                    if aj not in boo[ai]:
                        b12[ai].append(aj)
                        boo[ai].add(aj)
                    if ai not in boo[aj]:
                        b12[aj].append(ai)
                        boo[aj].add(ai)
            for _k, v in b12.items():
                v.sort()
        nb1x["12"] = b12

        b13 = {i: [] for i in range(natoms)}
        if nrexcl >= 2:
            for k, v in b12.items():
                for w in v:
                    for x in b12[w]:
                        if x not in boo[k]:
                            b13[k].append(x)
                            boo[k].add(x)
                        if k not in boo[x]:
                            b13[x].append(k)
                            boo[x].add(k)
            for _k, v in b13.items():
                v.sort()
        nb1x["13"] = b13

        b14 = {i: [] for i in range(natoms)}
        if nrexcl >= 3:
            for k, v in b13.items():
                for w in v:
                    for x in b12[w]:
                        if x not in boo[k]:
                            b14[k].append(x)
                            boo[k].add(x)
                        if k not in boo[x]:
                            b14[x].append(k)
                            boo[x].add(k)
            for _k, v in b14.items():
                v.sort()
        nb1x["14"] = b14

        return nb1x

    def get_atypes(self) -> list[str]:
        return [a.atype for a in self.atoms]

    def get_ptypes(self) -> list[str]:
        ta = TopoAtomTypes(self.system_uuid)
        return [ta.atomtypes[type_idx].particle_type for type_idx in self.atom_to_type_index]

    def get_charges(self) -> list[float]:
        ta = TopoAtomTypes(self.system_uuid)
        return [
            a.charge if hasattr(a, "charge") else ta.atomtypes[type_idx].charge
            for a, type_idx in zip(self.atoms, self.atom_to_type_index)
        ]

    def get_masses(self) -> list[float]:
        ta = TopoAtomTypes(self.system_uuid)
        return [
            a.mass if hasattr(a, "mass") else ta.atomtypes[type_idx].mass
            for a, type_idx in zip(self.atoms, self.atom_to_type_index)
        ]

    def get_vw(self) -> list[Tuple[float, float]]:
        ta = TopoAtomTypes(self.system_uuid)
        return [(ta.atomtypes[type_idx].V, ta.atomtypes[type_idx].W) for type_idx in self.atom_to_type_index]

    def update_charge(self,
                      partial_charges: Iterable[float] = None,
                      symm: dict[int, list[int]] = None,
                      to_round: bool = False):
        if to_round:
            assert self.allow_round_diff is not None
            assert self.allow_round_diff >= 0.
            pchgs = self.round_list_sum_to_int(self.get_charges() if partial_charges is None else partial_charges, symm,
                                               self.allow_round_diff)
        elif partial_charges is None:
            return
        else:
            pchgs = partial_charges

        assert self.natoms == len(pchgs)
        for pc, atom in zip(pchgs, self.atoms):
            if hasattr(atom, "charge"):
                atom.charge = pc
            else:
                setattr(atom, "charge", pc)

    @classmethod
    def from_records(cls, system_uuid, records: list[RecordText], sort_idx: bool, *,
                     round_on: Literal["", "r", "w", "rw"], allow_round_diff: float):
        """
        Exit on finding the second [ moleculetype ].
        """

        tm = cls(system_uuid, round_on=round_on, allow_round_diff=allow_round_diff)

        for r in records:
            if isinstance(r, RecordMoleculeType):
                if tm.moleculetype is None:
                    tm.moleculetype = deepcopy(r)
                else:
                    break
            elif isinstance(r, RecordAtom):
                tm.add_record_atom(r)
            elif isinstance(r, RecordBond):
                if r.verbose is None:
                    r.verbose = True
                tm.bonds.append(r)
            elif isinstance(r, RecordAngle):
                if r.verbose is None:
                    r.verbose = True
                tm.angles.append(r)
            elif isinstance(r, RecordDihedral):
                if r.verbose is None:
                    r.verbose = True
                tm.dihedrals.append(r)
            elif isinstance(r, RecordSettle):
                tm.settles.append(r)
            elif isinstance(r, RecordExclusion):
                tm.exclusions.append(r)
            elif isinstance(r, RecordPair):
                if r.verbose is None:
                    r.verbose = True
                tm.pairs.append(r)
            elif isinstance(r, RecordVirtualSite1):
                tm.virtual_sites1.append(r)
            elif isinstance(r, RecordVirtualSite2):
                tm.virtual_sites2.append(r)
            elif isinstance(r, RecordVirtualSite3):
                tm.virtual_sites3.append(r)
            elif isinstance(r, RecordVirtualSite4):
                tm.virtual_sites4.append(r)

        if sort_idx:
            tm._sort_bonds()
            tm._sort_angles()
            tm._sort_dihedrals()
            tm._sort_exclusions()
            tm._sort_pairs()

        # tm._check_pair_exclusion()
        if "r" in tm.round_on:
            if allow_round_diff is None or allow_round_diff < 0:
                raise RuntimeError(f"Incorrect allow_round_diff (current: {allow_round_diff}).")
            elif allow_round_diff > 0:
                tm.update_charge(partial_charges=None, symm=None, to_round=True)
            # else: pass

        return tm


########################
# full system topology #
########################


class TopoFullSystem:

    def __del__(self):
        TopoDefaults.remove(self.uuid)
        TopoAtomTypes.remove(self.uuid)

    def __init__(self):
        """
        e.g.
        mol_topos:
            [Water, Na, Cl, Protein, Ligand]
        molecules:
            [Ligand*1, Protein*1, Na*3, Cl*4, Water*1000]
        mol_to_topo_index:
            [4, 3, 1, 2, 0]
        """
        self.remarks: list[RecordText] = []
        self.mol_topos: list[TopoMolecule] = []
        self.system: RecordSystem = RecordSystem.from_text("Full System")
        self.molecules: list[RecordMolecule] = []

        self.mol_to_topo_index: list[int] = []
        self.uuid = uuid4()

        TopoDefaults(self.uuid)
        TopoAtomTypes(self.uuid)

    @classmethod
    def from_file(cls,
                  path: str,
                  incdir: str = None,
                  sort_idx: bool = True,
                  allow_unknown: bool = False,
                  *,
                  round_on: Literal["", "r", "w", "rw"] = "w",
                  allow_round_diff: float = 0.05):
        records = Records.from_file(path=path, incdir=incdir, allow_unknown=allow_unknown)
        return cls.from_records(records=records.all,
                                sort_idx=sort_idx,
                                round_on=round_on,
                                allow_round_diff=allow_round_diff)

    @classmethod
    def from_records(cls,
                     records: list[RecordText],
                     sort_idx: bool,
                     *,
                     round_on: Literal["", "r", "w", "rw"] = "w",
                     allow_round_diff: float = 0.05):
        tfs = cls()
        for r in records:
            if type(r) == RecordText:
                if r.text == "" and r.comment != "":
                    tfs.remarks.append(r)
            else:
                break

        td = TopoDefaults(tfs.uuid)
        ta = TopoAtomTypes(tfs.uuid)
        for r in records:
            if isinstance(r, RecordDefaults):
                td.add_record(r)
            elif isinstance(r, RecordAtomType):
                ta.add_record(r)
            elif isinstance(r, RecordSystem):
                tfs.system = deepcopy(r)
            elif isinstance(r, RecordMolecule):
                tfs.molecules.append(r)

        for idx, r in enumerate(records):
            if isinstance(r, RecordMoleculeType):
                m = TopoMolecule.from_records(system_uuid=tfs.uuid,
                                              records=records[idx:],
                                              sort_idx=sort_idx,
                                              round_on=round_on,
                                              allow_round_diff=allow_round_diff)
                for im in tfs.mol_topos:
                    if m.name == im.name:
                        raise TopoDuplicateNameMoleculeTypeException(m.name)
                tfs.mol_topos.append(m)

        if len(tfs.molecules) == 0:
            assert len(tfs.mol_topos)
            for mol_topo in tfs.mol_topos:
                tfs.molecules.append(RecordMolecule.from_text(f"{mol_topo.name} 1"))

        for mol in tfs.molecules:
            for idx, mol_topo in enumerate(tfs.mol_topos):
                if mol.name == mol_topo.name:
                    tfs.mol_to_topo_index.append(idx)

        return tfs

    def _strs_mol_atp_itp(self, idx):
        """
        atp
            [ atomtypes ]
        itp
            [ moleculetypes ]
            [ atoms ]
            ...
        """

        idx = [idx] if isinstance(idx, int) else [i for i in idx]
        ta = TopoAtomTypes(self.uuid)

        pos, itp_lines = [], []
        for ii in idx:
            mol_idx = self.mol_to_topo_index[ii]
            tm = self.mol_topos[mol_idx]
            itp_lines.append(str(tm))
            pos += list(set(tm.atom_to_type_index))
        pos.sort()

        return ta.str_slice(pos), "\n".join(itp_lines)

    def str_mol_itp(self, idx) -> str:
        """
        ; remarks
        [ atomtypes ]
        [ moleculetypes ]
        [ atoms ]
        ...
        """

        atp_line, itp_line = self._strs_mol_atp_itp(idx)
        lines = []

        for a in self.remarks:
            lines.append(a.comment)
        lines.append(atp_line)
        lines.append(itp_line)
        return "\n".join(lines)

    def write_itp(self, itp_path: str, idx=0):
        itp_str = self.str_mol_itp(idx)
        if os.path.exists(itp_path):
            abspath = os.path.abspath(itp_path)
            logger.warning(f"File {abspath} exists and will be overwritten.")
        logger.info(f"Writing itp: {itp_path}")
        with open(itp_path, "w") as fw:
            fw.write(itp_str)
            fw.write("\n")

    def str_system_top(self) -> str:
        """
        ; remarks
        [ defaults ]
        [ atomtypes ]
        ...
        [ system ]
        [ molecules ]
        """

        ta = TopoAtomTypes(self.uuid)
        lines = []

        for a in self.remarks:
            lines.append(a.comment)
        lines.append(str(TopoDefaults(self.uuid)))
        lines.append(str(ta))
        for a in self.mol_topos:
            lines.append(str(a))

        lines.append("")
        section = "system"
        lines.append(str(RecordSection(section=section)))
        lines.append(str(self.system))

        lines.append("")
        section = "molecules"
        lines.append(str(RecordSection(section=section)))
        lines.append(self.molecules[0].annotate())
        for a in self.molecules:
            lines.append(str(a))

        return "\n".join(lines)

    def write_top(self, top_path: str):
        top_str = self.str_system_top()
        if os.path.exists(top_path):
            abspath = os.path.abspath(top_path)
            logger.warning(f"File {abspath} exists and will be overwritten.")
        logger.info(f"Writing top: {top_path}")
        with open(top_path, "w") as fw:
            fw.write(top_str)
            fw.write("\n")

    def strs_system_top_atp_itp(self, atps: list[str], itps: list[str], mols: list[list[int]]):
        assert len(atps) == len(mols) and len(itps) == len(mols)

        # top string
        lines = []

        for a in self.remarks:
            lines.append(a.comment)
        lines.append(str(TopoDefaults(self.uuid)))

        lines.append("")
        for a in atps:
            lines.append(f'#include "{a}"')
        lines.append("")
        for a in itps:
            lines.append(f'#include "{a}"')

        lines.append("")
        section = "system"
        lines.append(str(RecordSection(section=section)))
        lines.append(str(self.system))

        lines.append("")
        section = "molecules"
        lines.append(str(RecordSection(section=section)))
        lines.append(self.molecules[0].annotate())
        for a in self.molecules:
            lines.append(str(a))

        # atp strings and atp strings
        atp_lines, itp_lines = [], []
        for idx in mols:
            atp, itp = self._strs_mol_atp_itp(idx=idx)
            atp_lines.append(atp)
            itp_lines.append(itp)

        return "\n".join(lines), atp_lines, itp_lines

    def write_top_atp_itp(self, top_path: str, atps: list[str], itps: list[str], mols: list[list[int]]):
        """
        Example

        [ molecules ]
        LIGAND         1
        PROTEIN        1
        WATER      10494
        NA            33
        CL            40
        CA             2

        tfs = TopoFullSystem.from_file(path=top_path)
        tfs.write_top_atp_itp(top_path="system.top",
            atps=["protein.atp", "water.atp", "ions.atp", "ligand.atp"],
            itps=["protein.itp", "water.itp", "ions.itp", "ligand.itp"],
            mols=[[1],           [2],         [3,4,5],    [0]])
        # Na, Cl, and Ca are written to "ions.atp" and "ions.itp".
        """
        top_str, atp_strs, itp_strs = self.strs_system_top_atp_itp(atps, itps, mols)
        for s, p in zip([top_str] + atp_strs + itp_strs, [top_path] + atps + itps):
            abspath = os.path.abspath(p)
            if os.path.exists(p):
                logger.warning(f"File {abspath} exists and will be overwritten.")
            with open(p, "w") as fw:
                fw.write(s)
                fw.write("\n")
