from newclid import GeometricSolverBuilder, GeometricSolver
from newclid.jgex.problem_builder import JGEXProblemBuilder
import numpy as np
from numpy.random import Generator as RngGenerator
from newclid.api_defaults import APIDefault
from newclid.rng import setup_rng
from newclid.symbols.points_registry import Point
from newclid.problem import (
    ProblemSetup,
    PredicateConstruction,
    Predicate,
)
from newclid.predicate_types import PredicateArgument
from newclid.numerical.geometries import PointNum
from IPython.display import HTML
from newclid.animation import html_animation
from pathlib import Path


from newclid.agent.follow_deductions import (
    ARPremiseConstruction,
    CachedARDeduction,
    CachedDeduction,
    CachedNumericalCheckDeduction,
    CachedReflexivityDeduction,
    CachedRuleDeduction,
    DeductionProvider,
    DeductionType,
)
from typing import TYPE_CHECKING, Any, Generator, Optional, Sequence, TypeVar
from newclid.agent.follow_deductions import DeductionType, FollowDeductions
from py_yuclid.yuclid_adapter import YuclidAdapter
import argparse

def p(name: str) -> PredicateArgument:
    return PredicateArgument(name)

def test_exist_problem_run():
  # Set the random generator
  rng = np.random.default_rng()

  # Build the problem setup from JGEX string
  problem_setup = JGEXProblemBuilder(rng=rng).with_problem_from_txt(
    "a b c = triangle a b c; "
    "d = on_tline d b a c, on_tline d c a b; "
    "e = on_line e a c, on_line e b d "
    "? perp a d b c"
  ).build()

  # We now build the solver on the problem
  solver: GeometricSolver = GeometricSolverBuilder().build(problem_setup)

  # And run the solver
  success = solver.run()

  if success:
      print("Successfuly solved the problem! Proof:")
  else:
      print("Failed to solve the problem...")

  print(f"Run infos {solver.run_infos}")

def test_generate_theorem():
  # 1. 初始化随机生成器（兼容类的 RngGenerator 要求）
  rng = setup_rng(42)  # 或直接传 int：rng=42

  # 2. 定义 ProblemSetup 所需的必填字段
  # 2.1 定义点（points）：几何元素的基础
  A = Point(name=p("a"), num=PointNum(x=1, y=0))  # 或 Point(id="A")，根据Point类的实际参数调整
  B = Point(name=p("b"), num=PointNum(x=0, y=1))
  C = Point(name=p("c"), num=PointNum(x=2, y=1))
  D = Point(name=p("d"), num=PointNum(x=1, y=1))
  points = [A, B, C, D]  # points字段必须是Point实例列表

  # 2.2 定义假设（assumptions）：即原题的 context 约束
  assumptions = [
    PredicateConstruction.from_str("ncoll a b c"),          # ABC是三角形
    PredicateConstruction.from_str("cong a b a c"),        # AB=AC（等腰）
    PredicateConstruction.from_str("midp d b c")          # D是BC中点
  ]

  # 2.3 定义目标（goals）
  goals = [PredicateConstruction.from_str("simtrir a d b a d c"),
           PredicateConstruction.from_str("perp a d b c")]

  # 3. 构造 ProblemSetup（严格按 Pydantic 要求传参）
  problem_setup = ProblemSetup(
      points=points,
      assumptions=assumptions,
      goals=goals
  )
  builder = GeometricSolverBuilder(rng=rng)
  solver = builder.build(problem_setup)
  success = solver.run()

  if success:
    # 提取推导的新特性
    derived_features = solver.proof()
    # HTML(html_animation(solver.animate()))
    print(derived_features)

def test_diagram_generate():
  rng = np.random.default_rng()

  # Build the problem setup from JGEX string
  problem_builder = JGEXProblemBuilder(rng=rng).with_problem_from_txt(
    "a b c = triangle a b c; "
    "d = on_tline d b a c, on_tline d c a b; "
    "e = on_line e a c, on_line e b d "
  )
  problem_setup = problem_builder.build()

  # We now build the solver on the problem
  solver: GeometricSolver = GeometricSolverBuilder().build(problem_setup)

  # And run the solver
  success = solver.run()
  print(success)
  jgex_problem = None
  if isinstance(problem_builder, JGEXProblemBuilder):
      jgex_problem = problem_builder.jgex_problem

  # Write figures to a local artifacts directory inside the repo
  out_dir = Path.cwd() / "artifacts"
  out_dir.mkdir(parents=True, exist_ok=True)

  solver.draw_figure(
        out_file= out_dir / "initial_figure.svg",
        jgex_problem=jgex_problem,
  )

def test_herui_29():
  # Set the random generator
  rng = setup_rng(42)
  he_adapter = YuclidAdapter()
  agent = FollowDeductions(he_adapter)

  # Build the problem setup from JGEX string
  problem_builder = JGEXProblemBuilder(rng=rng).with_problem_from_txt(
    "a b = segment a b; "
    "c = aconst a b a c 85o, aconst a b b c 40o; "
    "e = on_line e b c; "
    "d = on_pline0 d e a b, on_pline0 d a b e; "
    "f = on_pline0 f c a d, on_pline0 f d a c"  # NOTE 得用平行定义 
    # NOTE 点的定义就必须包含全部predicate 不然随意赋值coordinate后面的predicate check会fail 需要题目里面所有点的定义 都能被definition里面的函数定义出来
  )
  '''
  NOTE 用线段比例eqratio定义是出来的圆 交点比较奇怪 如图29_fig_wrong 符合条件 但是对不上
  "a b = segment a b; "
  "c = aconst a b a c 85o, aconst a b b c 40o; "
  "e = on_line e b c; "
  "f = on_line f b c, eqratio f a b b e a b c; "
  "d = eqratio d b c a b b c e, eqratio d b c a c b c f, on_pline0 d e a b, on_pline0 d f a c"  
  '''
  problem_setup = problem_builder.build()

  new_assumption = (
    PredicateConstruction.from_str("coll e c f"),
    PredicateConstruction.from_str("cong a b d e"),
    PredicateConstruction.from_str("cong a c d f"),
    PredicateConstruction.from_str("cong b e c f"),
    PredicateConstruction.from_str("eqratio a b a b c f b e")  # NOTE 这个其实跟f定义里面有一句是等价的 但是框架理解不了 这里还需要显式加进去
  )
  goals = ( PredicateConstruction.from_str("aconst d e d f 85o"), 
           PredicateConstruction.from_str("aconst d e e f 40o"), 
          )

  new_prob = problem_setup.with_new(new_assumptions = new_assumption, new_goals=goals)
  solver = (
    GeometricSolverBuilder(rng=rng)
    .with_deductive_agent(agent)
    .build(new_prob)
  )

  solver.run()
  proof = solver.proof()
  print(proof)

if __name__ == "__main__":
  pass
