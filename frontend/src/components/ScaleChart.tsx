import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

// only the pieces the scaling charts need - the full bundle is ~1MB
echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  CanvasRenderer,
])

/**
 * A thin ECharts wrapper.
 *
 * The chart is created inside `useEffect`, so it never runs during server-side
 * rendering - which is what lets `npm run check:render` verify these views
 * headlessly in an environment where a real browser cannot start.
 */
export default function ScaleChart({
  option,
  height = 260,
}: {
  option: Record<string, unknown>
  height?: number
}) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!hostRef.current) return
    const chart = echarts.init(hostRef.current)
    chartRef.current = chart
    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      chart.dispose()
      chartRef.current = null
    }
  }, [])

  useEffect(() => {
    chartRef.current?.setOption(option, true)
  }, [option])

  return <div className="chart" ref={hostRef} style={{ height }} />
}
