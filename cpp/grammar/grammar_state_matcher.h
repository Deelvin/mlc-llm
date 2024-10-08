/*!
 *  Copyright (c) 2023 by Contributors
 * \file grammar/grammar_state_matcher.h
 * \brief The header for the support of matching tokens to BNF grammar. This is the core
 * logic of the grammar-guided generation.
 */

#ifndef MLC_LLM_GRAMMAR_GRAMMAR_STATE_MATCHER_H_
#define MLC_LLM_GRAMMAR_GRAMMAR_STATE_MATCHER_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../support/encoding.h"
#include "grammar.h"
#include "support.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief A stateful matcher to match tokens to the specified BNF grammar. This class is the core
 * logic of the grammar-guided generation.
 *
 * \details This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm
 * to match characters to a BNF grammar. It keep track of the current state of the matching process
 * by maintaining several stacks internally as possible paths in the NPDA. It also supports
 * backtracking.
 *
 * It is particularly capable of finding the set of tokens that are acceptable for the next step
 * and storing them in a bitmask. This aids in grammar-guided generation.
 *
 * \example
 * \code
 * Tokenizer tokenizer = ...;
 * auto init_ctx = GrammarStateMatcher::CreateInitContext(grammar,
 *                                                        tokenizer->PostProcessedTokenTable());
 * GrammarStateMatcher matcher(init_ctx, 10);
 * matcher->AcceptToken(67);
 *
 * // Construct a DLTensor with shape (tokenizer.GetVocabSize() + 31) / 32, and dtype uint32.
 * DLTensor next_token_bitmask = ...;
 * matcher->FindNextTokenBitmask(&next_token_bitmask);
 *
 * // Rollback is supported
 * matcher->Rollback(1);
 * \endcode
 */
class GrammarStateMatcherNode : public Object {
 public:
  /*!
   * \brief Accept one token and update the state of the matcher.
   * \param token_id The id of the token to accept.
   * \return Whether the token is accepted.
   * \note Termination state.
   * When the end of the main rule is reached, the matcher can only accept the stop token.
   * The matcher is terminated after accepting the stop token, i.e. no AcceptToken or
   * FindNextTokenMask operations can be performed. The termination state can be canceled
   * using Rollback().
   */
  virtual bool AcceptToken(int32_t token_id, bool verbose = false) = 0;

  /*!
   * \brief Find the set of tokens that are acceptable for the next step and store them in a
   * bitmask.
   * \param next_token_bitmask The bitmask to store the result. The bitmask must be pre-allocated,
   * and its shape needs to be (ceil(vocab_size, 32),), with a dtype of uint32.
   */
  virtual void FindNextTokenBitmask(DLTensor* next_token_bitmask) = 0;

  /*!
   * \brief Find the jump-forward string for jump-forward decoding. This is the longest string that
   will be valid according to the current syntax.
   * \note This method does not change the grammar state.
   */
  virtual std::string FindJumpForwardString() = 0;

  /*!
   * \brief Rollback the matcher to a previous state.
   * \param num_tokens The number of tokens to rollback. It cannot exceed the current number of
   * steps, nor can it exceed the specified maximum number of rollback steps.
   */
  virtual void Rollback(int num_tokens) = 0;

  /*! \brief Get the maximum number of rollback steps allowed. */
  virtual int MaxRollbackSteps() const = 0;

  /*!
   * \brief Check if the matcher has accepted the stop token and terminated.
   * \sa AcceptToken
   */
  virtual bool IsTerminated() const = 0;

  /*! \brief Reset the matcher to the initial state. */
  virtual void ResetState() = 0;

  /*! \brief Set the stop token ids, overriding the existing defaults ones. */
  virtual void SetStopTokenIds(const std::vector<int32_t>& stop_token_ids) = 0;

  static constexpr const char* _type_key = "mlc.grammar.GrammarStateMatcher";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarStateMatcherNode, Object);
};

/*!
 * \brief The init context of a GrammarStateMatcher. It contains the preprocessing results of the
 * grammar and tokenizer.
 */
class GrammarStateInitContext;

class GrammarStateMatcher : public ObjectRef {
 public:
  /*!
   * \brief Construct a GrammarStateMatcher from the preprocessing result of type
   * GrammarStateInitContext.
   * \param init_ctx The init context. It is obtained through
   * CreateInitContext as a result of preprocessing the grammar and tokenizer.
   */
  GrammarStateMatcher(std::shared_ptr<GrammarStateInitContext> init_ctx,
                      int max_rollback_steps = 0);

  /*!
   * \brief Specify a grammar and token_table to return their preprocessing results. These results
   * are used to construct a GrammarStateMatcher. They can be stored elsewhere for quick
   * construction of GrammarStateMatcher.
   * \param grammar The grammar that the matcher follows.
   * \param token_table The tokens that the matcher requires for matching.
   */
  static std::shared_ptr<GrammarStateInitContext> CreateInitContext(
      const BNFGrammar& grammar, const std::vector<std::string>& token_table);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GrammarStateMatcher, ObjectRef, GrammarStateMatcherNode);
};

/*!
 * \brief A cache to get the grammar state init context for grammar or schema. This class avoids
 * redundant preprocessing of the grammar or schema when constructing a GrammarStateInitContext.
 * \note This class is associated with a token table when constructed. The token table is used to
 * create every grammar state init context. If multiple toke tables are used to create init
 * contexts, an instance of this class for each token table should be created.
 */
class GrammarInitContextCacheNode : public Object {
 public:
  /*! \brief Get the init context for pure JSON. */
  virtual std::shared_ptr<GrammarStateInitContext> GetInitContextForJSON() = 0;

  /*! \brief Get the init context for a JSON schema string. */
  virtual std::shared_ptr<GrammarStateInitContext> GetInitContextForJSONSchema(
      const std::string& schema) = 0;

  /*! \brief Clear the interal cache of init contexts. */
  virtual void Clear() = 0;

  static constexpr const char* _type_key = "mlc.serve.GrammarInitContextCacheNode";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarInitContextCacheNode, Object);
};

class GrammarInitContextCache : public ObjectRef {
 public:
  /*!
   * \brief Construct a GrammarInitContextCache with a token table. This class will always create
   * grammar state init contexts with this token table.
   * \param token_table The token table that the grammar will use.
   */
  GrammarInitContextCache(const std::vector<std::string>& token_table);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GrammarInitContextCache, ObjectRef,
                                        GrammarInitContextCacheNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_GRAMMAR_GRAMMAR_STATE_MATCHER_H_
