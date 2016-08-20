package water.rapids;

import water.*;
import water.fvec.*;


/** Sort the whole frame by the given columns
 */
public class ASTSort extends ASTPrim {
  @Override public String[] args() { return new String[]{"ary","cols"}; }
  @Override public String str(){ return "sort";}
  @Override int nargs() { return 1+2; } // (sort ary [cols])

  @Override public Val apply(Env env, Env.StackHelp stk, AST asts[]) {
    Frame fr = stk.track(asts[1].exec(env)).getFrame();
    int[] cols = asts[2].columns(fr.names());
    return new ValFrame(Merge.sort(fr,cols));
  }
}
