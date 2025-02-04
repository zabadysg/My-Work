# FEEDBACK PART

# if st.session_state.chat_history:
#     run_id=uuid.uuid5(
#                 uuid.NAMESPACE_DNS,
#                 str((len(st.session_state.chat_history) / 2) - 1) + st.session_state.user_id)
#     streamlit_feedback(
#         feedback_type="thumbs",
#         optional_text_label="[Optional] Please provide an explanation",
#         align="flex-start",
#         key="user_feedback",
#     )
#     if st.session_state.user_feedback:
#         client.create_feedback(
#             run_id=run_id,
#             key="User Feedback",
#             score=1.0 if st.session_state.user_feedback['score'] == "👍" else 0.0,
#             comment=st.session_state.user_feedback['text'],
#         )
