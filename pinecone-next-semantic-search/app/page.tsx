'use client' //Next.js 12 引入的一个新功能,  在 Next.js 中'use client'是一个特殊的指令，用于标识某个组件或函数应在客户端运行
import { send } from 'process'
import { useState } from 'react'

export default function Home() {

	const [queryValue, setQueryValue] = useState('query') // 设置数据库
	const [result, setResult] = useState('') // 保存 api 返回的结果
	const [loading, setLoading] = useState(false) // 加载 ui


	// 请求后端 API =>	read 跟 setup
	// 🌟 查询 Pinecone 数据库	
	async function sendQuery() {
		if(!queryValue) return // 如果没有查询内容, 直接返回 => 下面点击发送按钮后才会发起查询！
			setResult('')
			setLoading(true)
		try {
			const result = await fetch('/api/read', {
				method: 'POST',
				body: JSON.stringify(queryValue) // 🌟 请求体中中放入下方输入的内容
			})
			const json = await result.json()
			setResult(json.data) // 🔥保存查询回来的返回值
			setLoading(false)
		} catch(err) {
			console.log("请求 API 出错:", err);
			setLoading(false)
		}
	}

	// 🌟 传入创建 embedding 并传入 Pincone 数据库
	async function createIndexAndEmbeddings() {
		try {
			const result = await fetch('/api/setup', {
				method: "POST"
			})
			const json = await result.json()
			console.log("拿到 API 结果:", json);
		} catch(err) {
			console.log("请求 API 出错:", err);
		}
	}

	return (
		<main className="flex flex-col items-center justify-between p-24">
			<input
				className='text-black px-2 py-1'
				onChange={e => setQueryValue(e.target.value)} // 要查询的内容
			/>
			
			<button className="px-7 py-1 rounded-2xl bg-white text-black mt-2 mb-2" onClick={sendQuery}> 查询数据库 </button>

			{loading && <p>Asking AI ...</p>}
			{result && <p>{result}</p>}
			<button className="px-8 py-1 rounded-2xl bg-custom-blueColor text-white mt-2 mb-2" onClick={createIndexAndEmbeddings}> 创建 embedding </button>
		</main>
	)
}
