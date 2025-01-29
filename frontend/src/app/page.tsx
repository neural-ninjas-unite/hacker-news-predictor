'use client'
import { useState } from 'react'
import styles from './page.module.css'

export default function Home() {
  const [input, setInput] = useState('')
  const [score, setScore] = useState<number | null>(null)

  const handleSubmit = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/how_many_upvotes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          title: input,
          author: "anonymous",
          timestamp: new Date().toISOString()
        }),
      })
      const data = await response.json()
      setScore(data.upvotes)
    } catch (error) {
      console.error('Error:', error)
    }
  }

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter your title here..."
          className={styles.input}
        />
        <button onClick={handleSubmit} className={styles.button}>
          Get hacker news score!
        </button>
        {score !== null && (
          <div className={styles.result}>
            Predicted upvotes: {score}
          </div>
        )}
      </div>
    </main>
  )
} 