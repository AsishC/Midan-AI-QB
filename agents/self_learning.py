def capture_learning(db, question_id, field, ai_value, human_value):
    if ai_value == human_value:
        return
    db.execute('INSERT INTO ai_learning_signals (question_id, signal_type, before_value, after_value) VALUES (%s,%s,%s,%s)',
               (question_id, field, str(ai_value), str(human_value)))
