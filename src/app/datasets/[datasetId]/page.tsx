import AppFrame from "@/components/app-frame";
import Dashboard from "@/components/dashboard";

export default async function DatasetDetailRoute({ params, searchParams }: { params: Promise<{ datasetId: string }>; searchParams: Promise<{ view?: string }> }) {
  const [{ datasetId }, query] = await Promise.all([params, searchParams]);
  return <AppFrame><Dashboard datasetId={datasetId} initialTab={query.view}/></AppFrame>;
}
